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
"""

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor, LFMModality
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
    *,
    text: torch.Tensor,
    audio_in: torch.Tensor,
    audio_in_lens: torch.Tensor,
    audio_out: torch.Tensor,
    modality_flag: torch.Tensor,
    max_new_tokens: int = 20,
    text_temperature: float | None = None,
    text_top_k: int | None = None,
    audio_temperature: float | None = None,
    audio_top_k: int | None = None,
) -> Iterator[torch.Tensor]:
    """LFM2AudioModel.generate_interleaved に、ツールコール区間の扱いを足したもの

    `<|tool_call_start|>` を出したら `<|tool_call_end|>` まで音声を挟まずにテキストだけ生成し、
    ツールコールを閉じた時点でこのターンの生成を終える（続きはツール結果を入れてから再生成する）。
    """
    in_emb = model._prefill(
        text=text,
        audio_in=audio_in,
        audio_in_lens=audio_in_lens,
        audio_out=audio_out,
        modality_flag=modality_flag,
    )

    current_modality: LFMModality = LFMModality.TEXT
    modality_left: int = model.conf.interleaved_n_text
    cache = None

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
        cache = lfm_out.past_key_values

        if current_modality == LFMModality.TEXT:
            text_logits = nn.functional.linear(output_embeddings[0, -1], model.lfm.embed_tokens.weight)
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
            next_token = model._sample_audio_frame(
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
    *,
    threshold: float,
    max_new_tokens: int,
    text: torch.Tensor,
    audio_in: torch.Tensor,
    audio_in_lens: torch.Tensor,
    audio_out: torch.Tensor,
    modality_flag: torch.Tensor,
) -> tuple[float, list[torch.Tensor]]:
    """ツール判定パス（テキストのみ・greedy）

    先頭トークンが `<|tool_call_start|>` になる確率を返し、しきい値以上ならそのトークンを強制して
    `<|tool_call_end|>` まで生成したトークン列も返す。argmax だけで判定すると「大阪の天気」のような
    依頼でも呼ばないことが多いが、確率で見ると呼ぶべき依頼とそうでない依頼がはっきり分かれる。
    """
    in_emb = model._prefill(
        text=text,
        audio_in=audio_in,
        audio_in_lens=audio_in_lens,
        audio_out=audio_out,
        modality_flag=modality_flag,
    )
    cache = None
    prob = 0.0
    tokens: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        lfm_out = model.lfm(inputs_embeds=in_emb, past_key_values=cache, use_cache=True)
        cache = lfm_out.past_key_values
        text_logits = nn.functional.linear(lfm_out.last_hidden_state[0, -1], model.lfm.embed_tokens.weight)

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
    prefill_tokens: int
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
    # gr.Chatbot(type="messages") 形式の表示用ログ
    messages: list[dict[str, Any]] = field(default_factory=list)
    stats: list[PassStats] = field(default_factory=list)

    def context_breakdown(self) -> dict[str, int]:
        # 生成中に別スレッドから呼ばれるので、GPU 演算をせず shape だけで数える（MPS は同時利用で落ちる）
        total = self.chat.modality_flag.shape[1]
        text = self.chat.text.shape[1]
        audio_out = self.chat.audio_out.shape[1]
        return {"total": total, "text": text, "audio_in": total - text - audio_out, "audio_out": audio_out}


# "audio": 24kHz float32 の波形チャンク / "update": session.messages が更新された
Event = tuple[Literal["audio"], np.ndarray] | tuple[Literal["update"], None]


class Engine:
    def __init__(self, repo: str = HF_REPO, device: torch.device | None = None) -> None:
        self.device = device or pick_device()
        logger.info("Loading %s on %s", repo, self.device)
        model_dir = Path(snapshot_download(repo, ignore_patterns=["demo.mp4"]))
        self.proc = LFM2AudioProcessor.from_pretrained(model_dir, device=self.device).eval()
        self.model = LFM2AudioModel.from_pretrained(model_dir, device=self.device).eval()
        self.mimi = self.proc.mimi.eval()
        self.lock = threading.Lock()

        logger.info("Warmup tokenizer")
        with self.mimi.streaming(1), torch.no_grad():
            for _ in range(5):
                x = torch.randint(2048, (1, 8, 1), device=self.device)
                self.mimi.decode(x)

    def new_session(self) -> ChatSession:
        return ChatSession(chat=ChatState(self.proc))

    def count_tokens(self, text: str) -> int:
        return len(self.proc.text.encode(text, add_special_tokens=False))

    def decode_text(self, tokens: list[torch.Tensor]) -> str:
        if not tokens:
            return ""
        return self.proc.text.decode(torch.cat(tokens)).replace(TEXT_END, "")

    def run_user_turn(self, session: ChatSession, wav: torch.Tensor, sampling_rate: int, settings: GenSettings) -> Iterator[Event]:
        """ユーザーの音声 1 発話を受け取り、ツール実行を挟みながら assistant の応答を生成する"""
        with self.lock:
            yield from self._run_user_turn(session, wav, sampling_rate, settings)

    def _run_user_turn(self, session: ChatSession, wav: torch.Tensor, sampling_rate: int, settings: GenSettings) -> Iterator[Event]:
        tools = settings.tools()
        prompts = settings.prompts()
        system_prompt, route_system = prompts

        # システムプロンプトは会話の先頭にしか置けないので、変更されていたら会話をやり直す
        if session.prompts != prompts:
            if session.prompts is not None:
                session.messages.append(_info("システムプロンプト / ツール設定が変わったため、会話履歴をリセットしました"))
            session.chat = ChatState(self.proc)
            session.stats.clear()
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
        session.messages.append({"role": "user", "content": f"🎤 音声入力（{wav.shape[1] / sampling_rate:.1f} 秒）"})
        yield ("update", None)

        if route_system is not None:
            route_text = self._route(session, wav, sampling_rate, route_system, settings)
            yield from self._two_stage_turn(session, route_text, tools, settings)
        else:
            yield from self._native_turn(session, tools, settings)

    def _two_stage_turn(self, session: ChatSession, route_text: str, tools: list[dict[str, Any]], settings: GenSettings) -> Iterator[Event]:
        calls = self._report_calls(session, route_text)
        if calls:
            results = []
            for call in calls:
                result = execute_tool(call, tools)
                results.append({"tool": call.name, "result": result})
                session.messages.append(_tool_message(call, result, "user のテキストターンとして応答パスに渡した内容"))
            session.chat.new_turn("user")
            session.chat.add_text(TOOL_RESULT_PREFIX + json.dumps(results, ensure_ascii=False))
            session.chat.end_turn()
            yield ("update", None)

        session.chat.new_turn("assistant")
        yield from self._generate_pass(session, settings)

    def _native_turn(self, session: ChatSession, tools: list[dict[str, Any]], settings: GenSettings) -> Iterator[Event]:
        chat = session.chat
        for round_idx in range(settings.max_tool_rounds + 1):
            chat.new_turn("assistant")
            text = yield from self._generate_pass(session, settings)

            calls = self._report_calls(session, text)
            if not calls:
                break
            if round_idx == settings.max_tool_rounds:
                session.messages.append(_info(f"ツール呼び出しが {settings.max_tool_rounds} 回を超えたので打ち切りました", "⚠️ 打ち切り"))
                break

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
        prefill_tokens = int(chat.modality_flag.shape[1])

        start = time.perf_counter()
        prob, tokens = route_tool_call(self.model, threshold=settings.tool_threshold, max_new_tokens=ROUTE_MAX_TOKENS, **chat)
        elapsed = time.perf_counter() - start

        decision = "呼ぶ" if tokens else "呼ばない"
        session.stats.append(
            PassStats(
                kind="ツール判定",
                prefill_tokens=prefill_tokens,
                ttft=None,
                elapsed=elapsed,
                steps=max(len(tokens), 1),
                audio_seconds=0.0,
                note=f"p={prob:.4f}（しきい値 {settings.tool_threshold}）→ {decision}",
            )
        )
        return self.proc.text.decode(torch.cat(tokens)) if tokens else ""

    def _generate_pass(self, session: ChatSession, settings: GenSettings) -> Iterator[Event]:
        """assistant ターンを 1 回生成して chat に追記し、生成テキスト（特殊トークン込み）を返す"""
        chat = session.chat
        prefill_tokens = int(chat.modality_flag.shape[1])
        q: Queue[tuple[str, Any] | None] = Queue()

        def producer() -> None:
            try:
                with torch.no_grad(), self.mimi.streaming(1):
                    for t in generate_interleaved_with_tools(
                        self.model,
                        **chat,
                        max_new_tokens=settings.max_new_tokens,
                        # liquid_audio 側は temperature が float であることを assert している
                        text_temperature=float(settings.text_temperature) or None,
                        audio_temperature=float(settings.audio_temperature) or None,
                        audio_top_k=int(settings.audio_top_k) or None,
                    ):
                        # MPS はスレッド間で同時に使うと落ちるので、GPU テンソルはこのスレッドの外に出さない
                        q.put(("token", t.cpu()))
                        if t.numel() > 1 and not (t == AUDIO_EOS).any():
                            q.put(("wav", self.mimi.decode(t[None, :, None])[0].float().cpu().numpy().reshape(-1)))
            except Exception as e:  # noqa: BLE001 - 生成スレッドの例外は呼び出し側で再送出する
                q.put(("error", e))
            finally:
                q.put(None)

        session.messages.append({"role": "assistant", "content": "…"})
        message = session.messages[-1]

        start = time.perf_counter()
        ttft = 0.0
        out_text: list[torch.Tensor] = []
        out_audio: list[torch.Tensor] = []
        out_modality: list[LFMModality] = []
        audio_samples = 0

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        while (item := q.get()) is not None:
            kind, t = item
            if kind == "error":
                raise t
            if kind == "wav":
                audio_samples += t.shape[0]
                yield ("audio", t)
                continue

            if not ttft:
                ttft = time.perf_counter() - start
            if t.numel() == 1:
                out_text.append(t)
                out_modality.append(LFMModality.TEXT)
                message["content"] = _display_text(self.decode_text(out_text))
                yield ("update", None)
            else:
                out_audio.append(t)
                out_modality.append(LFMModality.AUDIO_OUT)
        thread.join()

        chat.append(
            text=torch.stack(out_text, 1) if out_text else chat.text.new_empty((1, 0)),
            audio_out=torch.stack(out_audio, 1) if out_audio else chat.audio_out.new_empty((chat.codebooks, 0)),
            modality_flag=torch.tensor(out_modality, dtype=chat.modality_flag.dtype),
        )
        chat.end_turn()

        stats = PassStats(
            kind="応答",
            prefill_tokens=prefill_tokens,
            ttft=ttft,
            elapsed=time.perf_counter() - start,
            steps=len(out_modality),
            audio_seconds=audio_samples / OUTPUT_SAMPLE_RATE,
        )
        session.stats.append(stats)
        logger.info("pass done: %s", stats)

        text = self.decode_text(out_text)
        message["content"] = _display_text(text) or "（テキスト出力なし）"
        yield ("update", None)
        return text


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
