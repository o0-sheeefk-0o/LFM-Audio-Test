"""LFM2.5-Audio-JP の読み込みと、システムプロンプト・ツールコール対応の speech-to-speech 生成

liquid_audio.demo の chat.py / model.py をベースに、以下を追加している。
  - CUDA 固定だったデバイスを自動選択（CUDA → MPS → CPU、環境変数 LFM_DEVICE で上書き可）
  - 任意のシステムプロンプトを system ターンに入れる
  - ツールコール（2 方式）
      native:    公式 chat_template 通り system にツール一覧を入れ、`<|tool_call_start|>...<|tool_call_end|>` を
                 検出したら実行して tool ターンで結果を返し、assistant ターンを再生成する
      two_stage: ツール判定パス（既定の指示なし + ツール一覧 + 今回の発話のみ）でツールコールだけ取り出し、
                 応答パス（既定の指示あり・ツール一覧なし・全履歴）には結果を user のテキストターンとして渡す
    実験では、system に `Respond with interleaved text and audio.` が無いとツールコールは出るが応答が崩れ、
    有るとツールを呼ばない。また tool ターンを入れると後続の応答が崩れるため、two_stage を既定にしている。
    判定は先頭トークンが `<|tool_call_start|>` になる確率のしきい値で行う（argmax だと取りこぼしが多い）。
  - interleaved 生成は「テキスト 6 トークン → 音声 9 フレーム」を強制的に交互に出すため、そのままだと
    ツールコールの途中に音声が割り込んで構文が壊れる。ツールコール区間だけはテキストのみで生成する
  - 前の返答の繰り返しへの対策。このモデルは会話履歴があると、「何か面白い話をして」のような自由回答の依頼に
    直前の返答（自己紹介や天気など）をそのまま返すことが多い（履歴なしなら正しく答える）。書き出しの
    テキスト（音声より先に出る 6 トークン）が履歴中の返答と同じなら打ち切り、履歴を捨てて答え直す
"""

import copy
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn

from accel import FastDepthformer, MimiWorker, Optimizations, load_cpu_mimi, quantize_int8
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality
from liquid_audio.utils import mel2emb_len
from tools import ToolCall, execute_tool, extract_tool_calls, load_tools

logger = logging.getLogger(__name__)

HF_REPO = os.environ.get("LFM_AUDIO_REPO", "LiquidAI/LFM2.5-Audio-1.5B-JP")
BASE_INSTRUCTION = "Respond with interleaved text and audio."
CONTEXT_LIMIT = 32_768
OUTPUT_SAMPLE_RATE = 24_000

TEXT_END = "<|text_end|>"
AUDIO_EOS = 2048

# テキストトークン ID（tokenizer_config.json の added_tokens_decoder）
IM_END_ID = 7
TOOL_CALL_START_ID = 10
TOOL_CALL_END_ID = 11
TEXT_END_ID = 130

TOOL_RESULT_PREFIX = "ツールの実行結果: "
ROUTE_MAX_TOKENS = 256
# 返答の書き出しのトークン数。interleaved 生成で最初の音声より前に出るテキスト（interleaved_n_text）と同じ長さ
REPEAT_CHECK_TOKENS = 6
# 書き出しがこれより短い返答（「はい。」など）は繰り返しとみなさない
REPEAT_MIN_TOKENS = 4

BasePosition = Literal["head", "tail", "none"]
ToolMode = Literal["two_stage", "native"]


def pick_device() -> torch.device:
    if name := os.environ.get("LFM_DEVICE"):
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_system_prompt(custom: str, base_position: BasePosition, tools: list[dict[str, Any]] | None) -> str:
    """system ターンの中身を組み立てる。ツール部分は chat_template.jinja と同じ書式"""
    custom = custom.strip()
    match base_position:
        case "head":
            parts = [BASE_INSTRUCTION, custom]
        case "tail":
            parts = [custom, BASE_INSTRUCTION]
        case "none":
            parts = [custom]
    prompt = "\n".join(p for p in parts if p)

    if tools:
        tool_list = ", ".join(json.dumps(t, ensure_ascii=False) for t in tools)
        prompt += ("\n" if prompt else "") + f"List of tools: <|tool_list_start|>[{tool_list}]<|tool_list_end|>"
    return prompt


@torch.no_grad()
def generate_interleaved_with_tools(
    model: LFM2AudioModel,
    in_emb: torch.Tensor,
    cache: Any = None,
    *,
    cache_out: dict[str, Any] | None = None,
    max_new_tokens: int = 20,
    text_temperature: float | None = None,
    text_top_k: int | None = None,
    audio_temperature: float | None = None,
    audio_top_k: int | None = None,
    sample_audio_frame: Callable[..., torch.Tensor] | None = None,
    text_head: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Iterator[torch.Tensor]:
    """LFM2AudioModel.generate_interleaved に、ツールコール区間の扱いを足したもの

    `<|tool_call_start|>` を出したら `<|tool_call_end|>` まで音声を挟まずにテキストだけ生成し、
    ツールコールを閉じた時点でこのターンの生成を終える（続きはツール結果を入れてから再生成する）。

    in_emb は未処理部分の入力埋め込み、cache はそれより前を処理済みの KV キャッシュ（prefix キャッシュ）。
    生成後のキャッシュは cache_out["cache"] に入る。
    sample_audio_frame / text_head で高速版の部品に差し替えられる（accel.py）。
    """
    sample_audio_frame = sample_audio_frame or model._sample_audio_frame
    text_head = text_head or (lambda h: nn.functional.linear(h, model.lfm.embed_tokens.weight))
    if cache_out is None:
        cache_out = {}

    current_modality: LFMModality = LFMModality.TEXT
    modality_left: int = model.conf.interleaved_n_text

    text_done: bool = False
    in_tool_call: bool = False

    for _ in range(max_new_tokens):
        modality_left -= 1
        lfm_out = model.lfm(
            inputs_embeds=in_emb,
            past_key_values=cache,
            use_cache=True,
        )
        output_embeddings = lfm_out.last_hidden_state
        # キャッシュはその場で更新されるので、どこで抜けても最新の状態が cache_out に残る
        cache = cache_out["cache"] = lfm_out.past_key_values

        if current_modality == LFMModality.TEXT:
            text_logits = text_head(output_embeddings[0, -1])
            next_token = model._sample_text_token(text_logits, temperature=text_temperature, top_k=text_top_k)

            if next_token == IM_END_ID:
                break

            yield next_token

            if next_token == TOOL_CALL_START_ID:
                in_tool_call = True
            if next_token == TOOL_CALL_END_ID:
                return
            if next_token == TEXT_END_ID:
                text_done = True
            if not in_tool_call and (modality_left <= 0 or text_done):
                current_modality = LFMModality.AUDIO_OUT
                modality_left = model.conf.interleaved_n_audio

            in_emb = model.lfm.embed_tokens(next_token)[None, :]

        elif current_modality == LFMModality.AUDIO_OUT:
            next_token = sample_audio_frame(
                output_embeddings[0, -1],
                temperature=audio_temperature,
                top_k=audio_top_k,
            )

            if not modality_left and not text_done:
                current_modality = LFMModality.TEXT
                modality_left = model.conf.interleaved_n_text

            if next_token[0] == AUDIO_EOS:
                next_token[:] = AUDIO_EOS
                current_modality = LFMModality.TEXT

            yield next_token
            in_emb = model.audio_embedding(next_token + model.codebook_offsets).sum(0)[None, None, :]


@torch.no_grad()
def route_tool_call(
    model: LFM2AudioModel,
    in_emb: torch.Tensor,
    cache: Any = None,
    *,
    threshold: float,
    max_new_tokens: int,
    text_head: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[float, list[torch.Tensor]]:
    """ツール判定パス（テキストのみ・greedy）

    先頭トークンが `<|tool_call_start|>` になる確率を返し、しきい値以上ならそのトークンを強制して
    `<|tool_call_end|>` まで生成したトークン列も返す。argmax だけで判定すると「大阪の天気」のような
    依頼でも呼ばないことが多いが、確率で見ると呼ぶべき依頼とそうでない依頼がはっきり分かれる。
    """
    text_head = text_head or (lambda h: nn.functional.linear(h, model.lfm.embed_tokens.weight))
    prob = 0.0
    tokens: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        lfm_out = model.lfm(inputs_embeds=in_emb, past_key_values=cache, use_cache=True)
        cache = lfm_out.past_key_values
        text_logits = text_head(lfm_out.last_hidden_state[0, -1])

        if step == 0:
            prob = text_logits.float().softmax(0)[TOOL_CALL_START_ID].item()
            if prob < threshold:
                return prob, []
            next_token = torch.tensor([TOOL_CALL_START_ID], device=text_logits.device)
        else:
            next_token = text_logits.argmax(keepdim=True)

        tokens.append(next_token.cpu())
        if next_token.item() in (TOOL_CALL_END_ID, IM_END_ID):
            break
        in_emb = model.lfm.embed_tokens(next_token)[None, :]

    return prob, tokens


@dataclass
class GenSettings:
    system_prompt: str = ""
    base_position: BasePosition = "head"
    tools_enabled: bool = False
    tools_json: str = "[]"
    tool_mode: ToolMode = "two_stage"
    # two_stage の判定で、先頭トークンが <|tool_call_start|> になる確率がこれ以上ならツールを呼ぶ
    tool_threshold: float = 0.005
    text_temperature: float = 0.0
    audio_temperature: float = 1.0
    audio_top_k: int = 4
    max_new_tokens: int = 1024
    max_tool_rounds: int = 3
    # 返答の書き出しが会話履歴中の返答と同じなら、履歴を捨てて system ＋今回の発話だけで答え直す
    repeat_fallback: bool = True

    def tools(self) -> list[dict[str, Any]]:
        return load_tools(self.tools_json) if self.tools_enabled else []

    def prompts(self) -> tuple[str, str | None]:
        """(会話履歴に入れる system, ツール判定パス用の system) を返す。後者は two_stage のときだけ"""
        tools = self.tools()
        if tools and self.tool_mode == "two_stage":
            return (
                build_system_prompt(self.system_prompt, self.base_position, None),
                build_system_prompt(self.system_prompt, "none", tools),
            )
        return build_system_prompt(self.system_prompt, self.base_position, tools), None


@dataclass
class PassStats:
    kind: Literal["ツール判定", "応答"]
    prefill_tokens: int  # このパスで新たに prefill した長さ（prefix キャッシュ済みの部分は含まない）
    context_tokens: int  # このパス開始時点のコンテキスト全体の長さ
    ttft: float | None
    elapsed: float
    steps: int
    audio_seconds: float
    note: str = ""


@dataclass
class ChatSession:
    chat: ChatState
    prompts: tuple[str, str | None] | None = None
    system_tokens: int = 0
    # prefix キャッシュ: chat の先頭 cached_len 位置ぶんを処理済みの KV キャッシュ
    lm_cache: Any = None
    cached_len: int = 0
    # gr.Chatbot(type="messages") 形式の表示用ログ
    messages: list[dict[str, Any]] = field(default_factory=list)
    stats: list[PassStats] = field(default_factory=list)
    # chat に入っている assistant の返答の書き出し（繰り返しの検知用）
    answer_heads: list[tuple[int, ...]] = field(default_factory=list)

    def context_breakdown(self) -> dict[str, int]:
        # 生成中に別スレッドから呼ばれるので、GPU 演算をせず shape だけで数える（MPS は同時利用で落ちる）
        total = self.chat.modality_flag.shape[1]
        text = self.chat.text.shape[1]
        audio_out = self.chat.audio_out.shape[1]
        return {"total": total, "text": text, "audio_in": total - text - audio_out, "audio_out": audio_out}


# "audio": 24kHz float32 の波形チャンク / "update": session.messages が更新された
Event = tuple[Literal["audio"], np.ndarray] | tuple[Literal["update"], None]


@dataclass
class TurnInput:
    """今回のターンの入力（繰り返しを検知して履歴なしで答え直すときに、chat を組み直すのに使う）"""

    wav: torch.Tensor
    sampling_rate: int
    notes: list[str] = field(default_factory=list)  # 発話の後に入れた user のテキストターン（ツールの結果）


@dataclass
class PassOutput:
    cache_out: dict[str, Any]
    text: list[torch.Tensor] = field(default_factory=list)
    audio: list[torch.Tensor] = field(default_factory=list)
    modality: list[LFMModality] = field(default_factory=list)
    audio_samples: int = 0
    ttft: float = 0.0


class Engine:
    def __init__(self, repo: str = HF_REPO, device: torch.device | None = None, opts: Optimizations | None = None) -> None:
        self.device = device or pick_device()
        self.opts = opts or Optimizations.from_env()
        logger.info("Loading %s on %s (optimizations: %s)", repo, self.device, self.opts.label())
        model_dir = Path(snapshot_download(repo, ignore_patterns=["demo.mp4"]))
        self.proc = LFM2AudioProcessor.from_pretrained(model_dir, device=self.device).eval()
        self.lock = threading.Lock()
        self._system_cache: dict[str, tuple[Any, int]] = {}

        # CPU で読み込んで（int8 ならここで量子化して）から GPU に移す。GPU 上で読み込むと、
        # 読み込み時の一時領域や量子化前の bf16 重みが MPS のメモリプールに残り、16GB 機ではスワップの原因になる
        self.model = LFM2AudioModel.from_pretrained(model_dir, device="cpu").eval()
        self.text_head = quantize_int8(self.model) if self.opts.int8 else None
        self.model.to(self.device)
        self.depthformer = FastDepthformer(self.model) if self.opts.fast_depthformer else None
        if self.opts.mimi_on_cpu:
            assert self.proc.mimi_weights_path is not None
            self.mimi = load_cpu_mimi(self.proc.mimi_weights_path)
        else:
            self.mimi = self.proc.mimi.eval()
        self.warmup()

    @torch.no_grad()
    def warmup(self) -> None:
        """MPS カーネルの初回コンパイルと torch.compile をここで済ませ、最初の発話を待たせないようにする"""
        start = time.perf_counter()
        settings = GenSettings(system_prompt="ウォームアップ", max_new_tokens=40)
        chat = ChatState(self.proc)
        chat.new_turn("user")
        chat.add_text("こんにちは")
        chat.end_turn()
        chat.new_turn("assistant")
        for kind, _ in self._stream_pass(self.embed(chat), None, {}, settings):
            pass
        route_tool_call(self.model, self.embed(chat), threshold=1.0, max_new_tokens=1, text_head=self.text_head)
        logger.info("Warmup done in %.1fs", time.perf_counter() - start)

    def new_session(self) -> ChatSession:
        return ChatSession(chat=ChatState(self.proc))

    def embed(self, chat: ChatState, start: int = 0) -> torch.Tensor:
        """chat の start 番目以降の位置の入力埋め込みを作る（start より前は KV キャッシュ済みの前提）

        音声入力は発話単位で conformer に通すので、キャッシュ済みの発話は再エンコードしない。
        """
        if start == 0:
            return self.model._prefill(**chat)
        head = chat.modality_flag[0, :start]
        n_text = int((head == LFMModality.TEXT).sum())
        n_audio_out = int((head == LFMModality.AUDIO_OUT).sum())
        n_audio_in = int((head == LFMModality.AUDIO_IN).sum())
        # キャッシュ済みの範囲は発話の境界で切れている（発話の途中までキャッシュされることはない）
        seg_ends = mel2emb_len(chat.audio_in_lens).cumsum(0).tolist()
        n_seg = sum(1 for end in seg_ends if end <= n_audio_in)
        assert n_audio_in == (seg_ends[n_seg - 1] if n_seg else 0), "prefix cache boundary splits an audio segment"
        mel_start = int(chat.audio_in_lens[:n_seg].sum())
        return self.model._prefill(
            text=chat.text[:, n_text:],
            audio_in=chat.audio_in[:, mel_start:],
            audio_in_lens=chat.audio_in_lens[n_seg:],
            audio_out=chat.audio_out[:, n_audio_out:],
            modality_flag=chat.modality_flag[:, start:],
        )

    def _system_prefix(self, system: str) -> tuple[Any, int]:
        """system ターンだけを処理済みの KV キャッシュ（ツール判定パスと、履歴なしでの答え直しで使い回す）

        呼び出し側はその場で更新されないよう deepcopy して使う。
        """
        cached = self._system_cache.get(system)
        if cached is None:
            chat = ChatState(self.proc)
            chat.new_turn("system")
            chat.add_text(system)
            chat.end_turn()
            out = self.model.lfm(inputs_embeds=self.embed(chat), use_cache=True)
            cached = self._system_cache[system] = (out.past_key_values, int(chat.modality_flag.shape[1]))
            # system プロンプトを変えるたびに増えないよう、直近のもの（判定用と応答用の 2 組）だけ残す
            for key in list(self._system_cache)[:-4]:
                del self._system_cache[key]
        return cached

    def count_tokens(self, text: str) -> int:
        return len(self.proc.text.encode(text, add_special_tokens=False))

    def decode_text(self, tokens: list[torch.Tensor]) -> str:
        if not tokens:
            return ""
        return self.proc.text.decode(torch.cat(tokens)).replace(TEXT_END, "")

    def run_user_turn(
        self, session: ChatSession, wav: torch.Tensor, sampling_rate: int, settings: GenSettings, audio_path: str | None = None
    ) -> Iterator[Event]:
        """ユーザーの音声 1 発話を受け取り、ツール実行を挟みながら assistant の応答を生成する

        audio_path を渡すと、会話ログにその音声ファイルの再生プレーヤーも出す（モデルが受け取った音声の確認用）。
        """
        with self.lock:
            yield from self._run_user_turn(session, wav, sampling_rate, settings, audio_path)

    def _run_user_turn(
        self, session: ChatSession, wav: torch.Tensor, sampling_rate: int, settings: GenSettings, audio_path: str | None
    ) -> Iterator[Event]:
        tools = settings.tools()
        prompts = settings.prompts()
        system_prompt, route_system = prompts

        # システムプロンプトは会話の先頭にしか置けないので、変更されていたら会話をやり直す
        if session.prompts != prompts:
            if session.prompts is not None:
                session.messages.append(_info("システムプロンプト / ツール設定が変わったため、会話履歴をリセットしました"))
            session.chat = ChatState(self.proc)
            session.lm_cache, session.cached_len = None, 0
            session.stats.clear()
            session.answer_heads.clear()
            if system_prompt:
                session.chat.new_turn("system")
                session.chat.add_text(system_prompt)
                session.chat.end_turn()
            session.prompts = prompts
            session.system_tokens = int(session.chat.text.shape[1])

        chat = session.chat
        chat.new_turn("user")
        chat.add_audio(wav, sampling_rate)
        chat.end_turn()
        duration = wav.shape[1] / sampling_rate
        logger.info("user utterance: %.2fs", duration)
        session.messages.append({"role": "user", "content": f"🎤 音声入力（{duration:.1f} 秒）"})
        if audio_path:
            session.messages.append({"role": "user", "content": {"path": audio_path, "mime_type": "audio/wav"}})
        yield ("update", None)

        turn = TurnInput(wav, sampling_rate)
        if route_system is not None:
            route_text = self._route(session, wav, sampling_rate, route_system, settings)
            yield from self._two_stage_turn(session, turn, route_text, tools, settings)
        else:
            yield from self._native_turn(session, turn, tools, settings)

    def _two_stage_turn(
        self, session: ChatSession, turn: TurnInput, route_text: str, tools: list[dict[str, Any]], settings: GenSettings
    ) -> Iterator[Event]:
        calls = self._report_calls(session, route_text)
        if calls:
            results = []
            for call in calls:
                result = execute_tool(call, tools)
                results.append({"tool": call.name, "result": result})
                session.messages.append(_tool_message(call, result, "user のテキストターンとして応答パスに渡した内容"))
            note = TOOL_RESULT_PREFIX + json.dumps(results, ensure_ascii=False)
            session.chat.new_turn("user")
            session.chat.add_text(note)
            session.chat.end_turn()
            turn.notes.append(note)
            yield ("update", None)

        session.chat.new_turn("assistant")
        yield from self._generate_pass(session, settings, turn)

    def _native_turn(self, session: ChatSession, turn: TurnInput, tools: list[dict[str, Any]], settings: GenSettings) -> Iterator[Event]:
        for round_idx in range(settings.max_tool_rounds + 1):
            session.chat.new_turn("assistant")
            # tool ターンを挟んだ後の返答は、履歴なしで組み直せないので答え直しの対象にしない
            text = yield from self._generate_pass(session, settings, turn if round_idx == 0 else None)

            calls = self._report_calls(session, text)
            if not calls:
                break
            if round_idx == settings.max_tool_rounds:
                session.messages.append(_info(f"ツール呼び出しが {settings.max_tool_rounds} 回を超えたので打ち切りました", "⚠️ 打ち切り"))
                break

            chat = session.chat  # 答え直しで組み直されていることがある
            for call in calls:
                result = execute_tool(call, tools)
                chat.new_turn("tool")  # type: ignore[arg-type]
                chat.add_text(f"<|tool_response_start|>{json.dumps(result, ensure_ascii=False)}<|tool_response_end|>")
                chat.end_turn()
                session.messages.append(_tool_message(call, result, "tool ターンとしてモデルに渡した内容"))
            yield ("update", None)

    def _report_calls(self, session: ChatSession, text: str) -> list[ToolCall]:
        calls, broken = extract_tool_calls(text)
        for body in broken:
            session.messages.append(_info(f"ツールコールをパースできませんでした:\n```\n{body}\n```", "⚠️ パース失敗"))
        return calls

    def _route(self, session: ChatSession, wav: torch.Tensor, sampling_rate: int, route_system: str, settings: GenSettings) -> str:
        """ツール判定パス。ツールを呼ぶと判定したときだけ、ツールコールのテキストを返す

        会話履歴は入れず、判定用 system + 今回の発話だけで判定する。履歴を入れると、
        ツールを呼ばずに答えている過去の assistant ターンに引っ張られて 2 ターン目以降ほとんど呼ばなくなる。
        """
        chat = ChatState(self.proc)
        chat.new_turn("system")
        chat.add_text(route_system)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(wav, sampling_rate)
        chat.end_turn()
        chat.new_turn("assistant")
        context_tokens = int(chat.modality_flag.shape[1])

        start = time.perf_counter()
        with torch.no_grad():
            if self.opts.prefix_cache:
                prefix_cache, prefix_len = self._system_prefix(route_system)
                cache = copy.deepcopy(prefix_cache)
            else:
                cache, prefix_len = None, 0
            in_emb = self.embed(chat, prefix_len)
            prob, tokens = route_tool_call(
                self.model, in_emb, cache, threshold=settings.tool_threshold, max_new_tokens=ROUTE_MAX_TOKENS, text_head=self.text_head
            )
        elapsed = time.perf_counter() - start

        decision = "呼ぶ" if tokens else "呼ばない"
        session.stats.append(
            PassStats(
                kind="ツール判定",
                prefill_tokens=int(in_emb.shape[1]),
                context_tokens=context_tokens,
                ttft=None,
                elapsed=elapsed,
                steps=max(len(tokens), 1),
                audio_seconds=0.0,
                note=f"p={prob:.4f}（しきい値 {settings.tool_threshold}）→ {decision}",
            )
        )
        return self.proc.text.decode(torch.cat(tokens)) if tokens else ""

    def _stream_pass(
        self, in_emb: torch.Tensor, cache: Any, cache_out: dict[str, Any], settings: GenSettings, stop: threading.Event | None = None
    ) -> Iterator[tuple[str, Any]]:
        """別スレッドで assistant ターンを生成し、("token", CPU テンソル) / ("wav", 波形) を順に返す

        stop をセットすると生成を打ち切る（それまでにキューに入った分は返る）。
        """
        q: Queue[tuple[str, Any] | None] = Queue()
        worker = MimiWorker(self.mimi, q) if self.opts.mimi_on_cpu else None
        sample_audio_frame = self.depthformer.sample if self.depthformer else None

        def producer() -> None:
            try:
                with torch.no_grad(), (nullcontext() if worker else self.mimi.streaming(1)):
                    for t in generate_interleaved_with_tools(
                        self.model,
                        in_emb,
                        cache,
                        cache_out=cache_out,
                        max_new_tokens=settings.max_new_tokens,
                        # liquid_audio 側は temperature が float であることを assert している
                        text_temperature=float(settings.text_temperature) or None,
                        audio_temperature=float(settings.audio_temperature) or None,
                        audio_top_k=int(settings.audio_top_k) or None,
                        sample_audio_frame=sample_audio_frame,
                        text_head=self.text_head,
                    ):
                        if stop is not None and stop.is_set():
                            break
                        # MPS はスレッド間で同時に使うと落ちるので、GPU テンソルはこのスレッドの外に出さない
                        t_cpu = t.cpu()
                        q.put(("token", t_cpu))
                        if t.numel() > 1 and not (t_cpu == AUDIO_EOS).any():
                            if worker:
                                worker.put(t_cpu)
                            else:
                                q.put(("wav", self.mimi.decode(t[None, :, None])[0].float().cpu().numpy().reshape(-1)))
            except Exception as e:  # noqa: BLE001 - 生成スレッドの例外は呼び出し側で再送出する
                q.put(("error", e))
            finally:
                if worker:
                    worker.close()
                q.put(None)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while (item := q.get()) is not None:
            if item[0] == "error":
                thread.join()
                raise item[1]
            yield item
        thread.join()

    def _generate_pass(self, session: ChatSession, settings: GenSettings, turn: TurnInput | None = None) -> Iterator[Event]:
        """assistant ターンを 1 回生成して chat に追記し、生成テキスト（特殊トークン込み）を返す

        turn を渡すと、書き出しが会話履歴中の返答と同じとき（前の返答の繰り返し）に生成を打ち切り、
        会話履歴を捨てて system ＋今回の入力だけで答え直す（会話はこのターンからやり直しになる）。
        書き出しは最初の音声より前に出るので、繰り返した部分は再生されない。
        """
        session.messages.append({"role": "assistant", "content": "…"})
        message = session.messages[-1]
        heads = session.answer_heads if settings.repeat_fallback and turn is not None else []
        note = ""

        start = time.perf_counter()
        if self.opts.prefix_cache and session.lm_cache is not None:
            cache, cached_len = session.lm_cache, session.cached_len
        else:
            cache, cached_len = None, 0
        context_tokens = int(session.chat.modality_flag.shape[1])
        with torch.no_grad():
            in_emb = self.embed(session.chat, cached_len)
        out = yield from self._collect_pass(in_emb, cache, settings, message, heads, start)

        if out is None:
            assert turn is not None and session.prompts is not None
            note = f"前の返答の繰り返しを検知 → 履歴なしで再生成（{time.perf_counter() - start:.2f} 秒ぶん破棄）"
            logger.info("repeated a previous answer; regenerating without history")
            session.messages.insert(len(session.messages) - 1, _info("前の返答を繰り返しかけたので、会話履歴を使わずに答え直しました（会話はこのターンからやり直し）", "🔁 答え直し"))
            message["content"] = "…"
            session.chat = self._fresh_chat(session.prompts[0], turn)
            session.answer_heads.clear()
            context_tokens = int(session.chat.modality_flag.shape[1])
            with torch.no_grad():
                if self.opts.prefix_cache and session.prompts[0]:
                    prefix_cache, cached_len = self._system_prefix(session.prompts[0])
                    cache = copy.deepcopy(prefix_cache)
                else:
                    cache, cached_len = None, 0
                in_emb = self.embed(session.chat, cached_len)
            out = yield from self._collect_pass(in_emb, cache, settings, message, [], start)
            assert out is not None

        chat = session.chat
        chat.append(
            text=torch.stack(out.text, 1) if out.text else chat.text.new_empty((1, 0)),
            audio_out=torch.stack(out.audio, 1) if out.audio else chat.audio_out.new_empty((chat.codebooks, 0)),
            modality_flag=torch.tensor(out.modality, dtype=chat.modality_flag.dtype),
        )
        chat.end_turn()
        if self.opts.prefix_cache and "cache" in out.cache_out:
            # 生成で流したトークンまでがキャッシュに入っている（chat の先頭からの位置と一致する）
            session.lm_cache = out.cache_out["cache"]
            session.cached_len = session.lm_cache.get_seq_length()
        head = _answer_head(out.text)
        if head is not None:
            session.answer_heads.append(head)

        stats = PassStats(
            kind="応答",
            prefill_tokens=int(in_emb.shape[1]),
            context_tokens=context_tokens,
            ttft=out.ttft,
            elapsed=time.perf_counter() - start,
            steps=len(out.modality),
            audio_seconds=out.audio_samples / OUTPUT_SAMPLE_RATE,
            note=note,
        )
        session.stats.append(stats)
        logger.info("pass done: %s", stats)

        text = self.decode_text(out.text)
        message["content"] = _display_text(text) or "（テキスト出力なし）"
        yield ("update", None)
        return text

    def _collect_pass(
        self,
        in_emb: torch.Tensor,
        cache: Any,
        settings: GenSettings,
        message: dict[str, Any],
        heads: list[tuple[int, ...]],
        start: float,
    ) -> Iterator[Event]:
        """1 回ぶんの生成を流しながら集めて PassOutput を返す。書き出しが heads のどれかと同じなら打ち切って None を返す"""
        out = PassOutput(cache_out={})
        stop = threading.Event()
        checked = not heads
        for kind, t in self._stream_pass(in_emb, cache, out.cache_out, settings, stop):
            if stop.is_set():
                continue  # 打ち切った後にキューに残っていた分は捨てる
            if kind == "wav":
                out.audio_samples += t.shape[0]
                yield ("audio", t)
                continue

            if not out.ttft:
                out.ttft = time.perf_counter() - start
            if t.numel() == 1:
                out.text.append(t)
                out.modality.append(LFMModality.TEXT)
                message["content"] = _display_text(self.decode_text(out.text))
                yield ("update", None)
            else:
                out.audio.append(t)
                out.modality.append(LFMModality.AUDIO_OUT)

            # 書き出しのテキストが出そろった時点（最初の音声フレームより前）で 1 回だけ調べる
            if not checked and (len(out.text) >= REPEAT_CHECK_TOKENS or t.numel() > 1):
                checked = True
                head = _answer_head(out.text)
                if head is not None and any(prev[: len(head)] == head for prev in heads):
                    stop.set()
        return None if stop.is_set() else out

    def _fresh_chat(self, system: str, turn: TurnInput) -> ChatState:
        """会話履歴なしの chat（system ＋今回の発話＋ツールの結果）"""
        chat = ChatState(self.proc)
        if system:
            chat.new_turn("system")
            chat.add_text(system)
            chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(turn.wav, turn.sampling_rate)
        chat.end_turn()
        for note in turn.notes:
            chat.new_turn("user")
            chat.add_text(note)
            chat.end_turn()
        chat.new_turn("assistant")
        return chat


def _answer_head(text: list[torch.Tensor]) -> tuple[int, ...] | None:
    """返答の書き出し。短すぎる返答とツールコールは繰り返しの判定に使わない"""
    head = tuple(int(t) for t in text[:REPEAT_CHECK_TOKENS])
    if len(head) < REPEAT_MIN_TOKENS or TOOL_CALL_START_ID in head:
        return None
    return head


def _display_text(text: str) -> str:
    # 特殊トークンが HTML として解釈されないようコード表記にする
    for tok in ("<|tool_call_start|>", "<|tool_call_end|>"):
        text = text.replace(tok, f"`{tok}`")
    return text


def _info(text: str, title: str = "ℹ️ 情報") -> dict[str, Any]:
    return {"role": "assistant", "content": text, "metadata": {"title": title}}


def _tool_message(call: ToolCall, result: dict[str, Any], how: str) -> dict[str, Any]:
    body = f"**呼び出し**\n```python\n{call}\n```\n**結果（{how}）**\n```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"
    return {"role": "assistant", "content": body, "metadata": {"title": f"🔧 ツール実行: {call.name}"}}
