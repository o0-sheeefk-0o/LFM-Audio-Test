"""LFM2.5-Audio-1.5B-JP speech-to-speech デモ（システムプロンプト / ツールコール検証版）

liquid_audio.demo.chat をベースに、システムプロンプト入力・ツール定義・コンテキスト使用量の表示を追加したもの。
起動: uv run python app.py  →  http://localhost:7860
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from fastrtc import AdditionalOutputs, WebRTC

import vad
from engine import BASE_INSTRUCTION, CONTEXT_LIMIT, OUTPUT_SAMPLE_RATE, ChatSession, Engine, GenSettings
from filler import make_filler
from tools import SAMPLE_TOOLS_JSON
from vad import UtteranceReplyOnPause

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

engine = Engine()

# モデルが受け取ったユーザー音声の保存先（会話ログで再生できるようにする）。起動のたびに空にする
RECORDINGS = Path(__file__).resolve().parent / "outputs" / "recordings"
shutil.rmtree(RECORDINGS, ignore_errors=True)
RECORDINGS.mkdir(parents=True, exist_ok=True)

DEFAULT_SYSTEM_PROMPT = "あなたは親切な日本語の音声アシスタントです。回答は短く、話し言葉で答えてください。"


class SessionHolder:
    """gr.State に入れる入れ物。ChatState は重いので初回の発話時に作る"""

    session: ChatSession | None = None


def render_stats(session: ChatSession | None) -> str:
    if session is None:
        return "まだ会話はありません。"
    ctx = session.context_breakdown()
    lines = [
        f"**コンテキスト使用量**: {ctx['total']:,} / {CONTEXT_LIMIT:,} tokens（{ctx['total'] / CONTEXT_LIMIT:.1%}）  ",
        f"内訳: システムターン {session.system_tokens:,} ／ テキスト計 {ctx['text']:,} ／ 音声入力 {ctx['audio_in']:,} ／ 音声出力 {ctx['audio_out']:,}",
    ]
    if session.stats:
        lines += [
            "",
            "| # | パス | prefill（新規 / 全体） | 最初のトークンまで | 所要時間 | ステップ/秒 | 音声長 | RTF | 備考 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for i, s in list(enumerate(session.stats, 1))[-6:]:
            ttft = f"{s.ttft:.2f} 秒" if s.ttft is not None else "-"
            rtf = f"{s.elapsed / s.audio_seconds:.2f}" if s.audio_seconds else "-"
            lines.append(
                f"| {i} | {s.kind} | {s.prefill_tokens:,} / {s.context_tokens:,} | {ttft} | {s.elapsed:.2f} 秒 | {s.steps / s.elapsed:.1f} | {s.audio_seconds:.1f} 秒 | {rtf} | {s.note} |"
            )
        lines.append("\nRTF = 生成にかかった時間 ÷ 生成した音声の長さ（1 未満ならリアルタイムより速い）")
    return "\n".join(lines)


def render_system_info(system_prompt: str, base_position: str, tools_enabled: bool, tools_json: str, tool_mode: str) -> tuple[str, str]:
    settings = GenSettings(
        system_prompt=system_prompt,
        base_position=base_position,  # type: ignore[arg-type]
        tools_enabled=tools_enabled,
        tools_json=tools_json,
        tool_mode=tool_mode,  # type: ignore[arg-type]
    )
    try:
        prompt, route_prompt = settings.prompts()
    except Exception as e:
        return f"⚠️ ツール定義の JSON が不正です: {e}", ""

    def describe(label: str, prompt: str) -> tuple[str, str]:
        if not prompt:
            return f"**{label}**: なし（system ターン自体を入れません）", ""
        turn = f"<|im_start|>system\n{prompt}<|im_end|>\n"
        n = engine.count_tokens(turn)
        return f"**{label}**: {n:,} tokens ／ {len(prompt):,} 文字 ／ コンテキスト上限の {n / CONTEXT_LIMIT:.1%}", turn

    if route_prompt is None:
        info, turn = describe("システムターン", prompt)
        return info, turn
    info, turn = describe("応答パスの system", prompt)
    route_info, route_turn = describe("ツール判定パスの system", route_prompt)
    return f"{info}  \n{route_info}", f"===== 応答パス（会話履歴に入る）=====\n{turn}\n===== ツール判定パス =====\n{route_turn}"


def chat_response(
    audio: tuple[int, np.ndarray],
    _id: str,
    holder: SessionHolder,
    system_prompt: str,
    base_position: str,
    tools_enabled: bool,
    tools_json: str,
    tool_mode: str,
    tool_threshold: float,
    text_temperature: float,
    audio_temperature: float,
    audio_top_k: int,
    max_new_tokens: int,
    repeat_fallback: bool,
):
    if holder.session is None:
        holder.session = engine.new_session()
    session = holder.session

    settings = GenSettings(
        system_prompt=system_prompt,
        base_position=base_position,  # type: ignore[arg-type]
        tools_enabled=tools_enabled,
        tools_json=tools_json,
        tool_mode=tool_mode,  # type: ignore[arg-type]
        tool_threshold=tool_threshold,
        text_temperature=text_temperature,
        audio_temperature=audio_temperature,
        audio_top_k=audio_top_k,
        max_new_tokens=max_new_tokens,
        repeat_fallback=repeat_fallback,
    )

    rate, wav = audio
    audio_path = RECORDINGS / f"{time.strftime('%Y%m%d-%H%M%S')}-{len(session.messages):03d}.wav"
    sf.write(audio_path, wav.reshape(-1), rate)
    try:
        for kind, payload in engine.run_user_turn(
            session, torch.tensor(wav / 32_768, dtype=torch.float), rate, settings, audio_path=str(audio_path)
        ):
            if kind == "audio":
                yield (OUTPUT_SAMPLE_RATE, (payload * 32_767).astype(np.int16)[None, :])
            else:
                yield AdditionalOutputs(list(session.messages), render_stats(session))
    except Exception as e:
        logger.exception("generation failed")
        # 途中で落ちると ChatState が中途半端になるので作り直す（表示ログは残す）
        holder.session = engine.new_session()
        holder.session.messages = session.messages + [
            {"role": "assistant", "content": f"`{type(e).__name__}: {e}`\n\n会話履歴をリセットしました。", "metadata": {"title": "❌ エラー"}}
        ]
        yield AdditionalOutputs(list(holder.session.messages), render_stats(holder.session))


def clear(holder: SessionHolder) -> tuple[SessionHolder, list[Any], str]:
    gr.Info("Cleared chat history", duration=3)
    # fastrtc 側が古い参照を握っていても確実にリセットされるよう、同じオブジェクトを書き換える
    holder.session = None
    return holder, [], render_stats(None)


def add_filler(system_prompt: str, target_tokens: float) -> str:
    return system_prompt.rstrip() + make_filler(int(target_tokens), engine.count_tokens)


with gr.Blocks(title="LFM2.5-Audio-JP S2S") as demo:
    gr.Markdown(
        f"# LFM2.5-Audio-1.5B-JP speech-to-speech chat\n"
        f"マイクで話しかけると、無音を検出したタイミングで応答します。デバイス: `{engine.device}`  \n"
        "システムプロンプト・ツール設定は**次の発話時**に反映されます（変更されていた場合は会話履歴を自動でリセット）。"
    )

    holder_state = gr.State(SessionHolder())

    with gr.Row():
        with gr.Column(scale=5):
            webrtc = WebRTC(
                modality="audio",
                mode="send-receive",
                full_screen=False,
            )
            chatbot = gr.Chatbot(type="messages", label="会話ログ", height=480, show_copy_button=True)
            stats_md = gr.Markdown(render_stats(None))
            clear_btn = gr.Button("Reset chat")

        with gr.Column(scale=4):
            with gr.Accordion("システムプロンプト", open=True):
                system_prompt = gr.Textbox(
                    value=DEFAULT_SYSTEM_PROMPT,
                    label="システムプロンプト",
                    lines=8,
                    max_lines=20,
                )
                base_position = gr.Radio(
                    choices=[("先頭に付ける", "head"), ("末尾に付ける", "tail"), ("付けない", "none")],
                    value="head",
                    label=f"モデル既定の指示「{BASE_INSTRUCTION}」",
                    info="モデルカード推奨の一文。外すと応答テキストが崩れやすい（中国語混じり・繰り返し）。2 段階モードのツール判定パスでは常に外す",
                )
                system_info = gr.Markdown()
                with gr.Row():
                    filler_tokens = gr.Number(value=2000, minimum=100, maximum=CONTEXT_LIMIT, step=500, label="ダミー資料のトークン数", scale=2)
                    filler_btn = gr.Button("長さ検証用のダミー資料（社員名簿）を追記", scale=3)

            with gr.Accordion("ツール", open=True):
                tools_enabled = gr.Checkbox(value=True, label="ツールを渡す")
                tool_mode = gr.Radio(
                    choices=[("2 段階（ツール判定 → 応答）", "two_stage"), ("ネイティブ（公式テンプレート通り）", "native")],
                    value="two_stage",
                    label="ツールの扱い方",
                    info=(
                        "2 段階: 既定の指示なし + ツール一覧の system と今回の発話だけで、ツールを呼ぶかを判定し、"
                        "応答は既定の指示あり・ツール一覧なしの system で、結果を user のテキストとして渡して生成する。"
                        "ネイティブ: system にツール一覧、結果は tool ターン（応答が崩れやすい）"
                    ),
                )
                tool_threshold = gr.Slider(
                    0.0005, 0.5, value=0.005, step=0.0005,
                    label="ツール判定しきい値（2 段階のみ）",
                    info="判定パスの先頭トークンが <|tool_call_start|> になる確率がこれ以上ならツールを呼ぶ。各ターンの確率は下の統計表の備考に出ます",
                )
                tools_json = gr.Code(value=SAMPLE_TOOLS_JSON, language="json", label="ツール定義（編集可。実装のないツールはモック応答を返します）", lines=12, max_lines=24)
                reset_tools_btn = gr.Button("サンプルに戻す", size="sm")

            with gr.Accordion("音声入力（発話の切り出し）", open=False):
                gr.Markdown(
                    "話し終わりを検出してから応答します。言葉の途中の間で切れて質問が途中までしか届かないときは、"
                    "無音の長さを伸ばしてください。マイクの音量が小さいときは VAD のしきい値と最小音量を下げてください。"
                    "応答音声の再生中はマイク入力を受け付けません（応答の声や残響を発話として拾わないため）。"
                    "モデルが受け取った音声は会話ログで再生できます。"
                )
                end_silence = gr.Slider(0.3, 2.0, value=vad.VAD.end_silence_s, step=0.1, label="発話終了と判定する無音の長さ（秒）")
                vad_threshold = gr.Slider(0.1, 0.9, value=vad.VAD.threshold, step=0.05, label="VAD のしきい値（発話とみなす確率）")
                min_level = gr.Slider(-60, -20, value=vad.VAD.min_level_db, step=1, label="発話とみなす最小音量（dBFS。これより小さい音は残響・物音として無視）")

            with gr.Accordion("生成パラメータ", open=False):
                text_temperature = gr.Slider(0, 2, value=0, step=0.05, label="text temperature（0 = greedy）")
                audio_temperature = gr.Slider(0, 2, value=1.0, step=0.05, label="audio temperature（0 = greedy）")
                audio_top_k = gr.Slider(0, 64, value=4, step=1, label="audio top-k（0 = 無制限）")
                max_new_tokens = gr.Slider(64, 4096, value=1024, step=64, label="max new tokens（1 ターンあたり）")
                repeat_fallback = gr.Checkbox(
                    value=True,
                    label="前の返答を繰り返しかけたら、会話履歴を使わずに答え直す",
                    info="このモデルは履歴があると、自由回答の依頼に直前の返答（自己紹介など）をそのまま返しやすい。答え直すと会話はそのターンからやり直しになる",
                )

            with gr.Accordion("実際にモデルへ渡す system ターン", open=False):
                system_preview = gr.Textbox(show_label=False, lines=8, max_lines=20, interactive=False, show_copy_button=True)

    webrtc.stream(
        UtteranceReplyOnPause(
            chat_response,  # type: ignore[arg-type]
            input_sample_rate=24_000,
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            can_interrupt=False,
        ),
        inputs=[
            webrtc,
            holder_state,
            system_prompt,
            base_position,
            tools_enabled,
            tools_json,
            tool_mode,
            tool_threshold,
            text_temperature,
            audio_temperature,
            audio_top_k,
            max_new_tokens,
            repeat_fallback,
        ],
        outputs=[webrtc],
    )
    webrtc.on_additional_outputs(
        lambda messages, stats: (messages, stats),
        outputs=[chatbot, stats_md],
    )
    clear_btn.click(clear, inputs=[holder_state], outputs=[holder_state, chatbot, stats_md])

    info_inputs = [system_prompt, base_position, tools_enabled, tools_json, tool_mode]
    for component in info_inputs:
        component.change(render_system_info, inputs=info_inputs, outputs=[system_info, system_preview])
    demo.load(render_system_info, inputs=info_inputs, outputs=[system_info, system_preview])

    filler_btn.click(add_filler, inputs=[system_prompt, filler_tokens], outputs=[system_prompt])
    reset_tools_btn.click(lambda: SAMPLE_TOOLS_JSON, outputs=[tools_json])
    end_silence.change(lambda v: setattr(vad.VAD, "end_silence_s", float(v)), inputs=[end_silence])
    vad_threshold.change(lambda v: setattr(vad.VAD, "threshold", float(v)), inputs=[vad_threshold])
    min_level.change(lambda v: setattr(vad.VAD, "min_level_db", float(v)), inputs=[min_level])


def main():
    demo.launch(allowed_paths=[str(RECORDINGS)])


if __name__ == "__main__":
    main()
