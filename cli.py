"""マイクなしで speech-to-speech を検証する CLI

例:
  # macOS の say で作った音声を入力にして 2 ターン会話
  uv run python cli.py --say "東京の天気を教えて" --say "じゃあ傘はいるかな"

  # 長いシステムプロンプト（ダミー資料 8000 トークン）で、奥の情報を参照できるか試す
  uv run python cli.py --filler-tokens 8000 --say "社員番号1200の人の内線番号を教えて"

各ターンの応答音声は --out-dir（既定 outputs/）に保存される。
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from engine import OUTPUT_SAMPLE_RATE, Engine, GenSettings
from filler import make_filler
from tools import SAMPLE_TOOLS_JSON

DEFAULT_SYSTEM_PROMPT = "あなたは親切な日本語の音声アシスタントです。回答は短く、話し言葉で答えてください。"


def synthesize(text: str, voice: str) -> tuple[torch.Tensor, int]:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "in.wav"
        subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True)
        wav, sr = sf.read(path, dtype="float32")
    return torch.from_numpy(wav).unsqueeze(0), sr


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(wav.mean(1)).unsqueeze(0), sr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--say", action="append", default=[], help="macOS の say で合成してユーザー発話にする（複数指定で複数ターン）")
    parser.add_argument("--audio", action="append", default=[], help="ユーザー発話の音声ファイル（--say の後に続けて使う）")
    parser.add_argument("--voice", default="Kyoko", help="say の音声")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--system-prompt-file", help="システムプロンプトをファイルから読む")
    parser.add_argument("--filler-tokens", type=int, default=0, help="ダミー資料（社員名簿）をこのトークン数ぶん追記する")
    parser.add_argument("--base", choices=["head", "tail", "none"], default="head", help="既定の指示文の位置")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--tools-file", help="ツール定義 JSON（省略時はサンプル）")
    parser.add_argument("--tool-mode", choices=["two_stage", "native"], default="two_stage", help="ツールの扱い方")
    parser.add_argument("--tool-threshold", type=float, default=0.005, help="2 段階方式のツール判定しきい値")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    engine = Engine()

    system_prompt = Path(args.system_prompt_file).read_text() if args.system_prompt_file else args.system_prompt
    if args.filler_tokens:
        system_prompt = system_prompt.rstrip() + make_filler(args.filler_tokens, engine.count_tokens)

    settings = GenSettings(
        system_prompt=system_prompt,
        base_position=args.base,
        tools_enabled=not args.no_tools,
        tools_json=Path(args.tools_file).read_text() if args.tools_file else SAMPLE_TOOLS_JSON,
        tool_mode=args.tool_mode,
        tool_threshold=args.tool_threshold,
        max_new_tokens=args.max_new_tokens,
    )
    system, route_system = settings.prompts()
    print(f"system ターン: {engine.count_tokens(system):,} tokens / {len(system):,} 文字")
    if route_system is not None:
        print(f"ツール判定パスの system: {engine.count_tokens(route_system):,} tokens")

    utterances = [synthesize(t, args.voice) for t in args.say] + [load_wav(p) for p in args.audio]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    session = engine.new_session()
    for turn, (wav, sr) in enumerate(utterances, 1):
        printed, passes = len(session.messages), len(session.stats)
        chunks: list[np.ndarray] = []
        for kind, payload in engine.run_user_turn(session, wav, sr, settings):
            if kind == "audio":
                chunks.append(payload)

        for m in session.messages[printed:]:
            title = (m.get("metadata") or {}).get("title")
            print(f"\n[{m['role']}{' / ' + title if title else ''}]\n{m['content']}")

        if chunks:
            path = out_dir / f"turn{turn}.wav"
            sf.write(path, np.concatenate(chunks), OUTPUT_SAMPLE_RATE)
            print(f"\n→ 応答音声: {path}")

        ctx = session.context_breakdown()
        print(f"→ コンテキスト: {json.dumps(ctx)}")
        for s in session.stats[passes:]:
            ttft = f" / TTFT {s.ttft:.2f}s" if s.ttft is not None else ""
            print(f"→ {s.kind}: prefill {s.prefill_tokens:,} / {s.context_tokens:,} tok{ttft} / {s.steps} steps in {s.elapsed:.1f}s / 音声 {s.audio_seconds:.1f}s {s.note}")


if __name__ == "__main__":
    main()
