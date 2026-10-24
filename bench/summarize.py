"""bench/results の計測結果を表にして表示する

  uv run python bench/summarize.py baseline opt1-depthformer opt2-mimi-cpu opt3-int8
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(label: str, mode: str) -> dict | None:
    path = RESULTS / f"{label}-{mode}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    labels = sys.argv[1:]
    print("## 1 ステップの内訳（ms、GPU 同期あり）\n")
    print("| 段階 | バックボーン | テキストヘッド | 音声フレーム | Mimi | テキスト 1 step | 音声 1 step | 見積もり RTF | 起動 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        d = load(label, "components")
        if not d:
            continue
        c, e = d["data"], d["data"]["estimate"]
        print(
            f"| {label} | {c['lfm_step']['latency_ms']:.1f} | {c['text_head']['latency_ms']:.1f} | {c['audio_frame']['latency_ms']:.1f} | "
            f"{c['mimi_decode']['latency_ms']:.1f} | {e['text_step_ms']:.1f} | {e['audio_step_ms']:.1f} | {e['rtf']:.2f} | {d['env']['load_s']:.0f} 秒 |"
        )

    print("\n## 1 ターン全体（3 回の中央値）\n")
    print("| 段階 | ケース | 最初の音声まで | うちツール判定 | 応答の最初のトークンまで | step/s | RTF | ターン全体 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        d = load(label, "e2e")
        if not d:
            continue
        for case, v in d["data"].items():
            if case == "cold_first_turn":
                continue
            print(
                f"| {label} | {case} | {v['first_audio_s']:.2f} 秒 | {v['route_s']:.2f} 秒 | {v['answer_ttft_s']:.2f} 秒 | "
                f"{v['steps_per_s']:.1f} | {v['rtf']:.2f} | {v['total_s']:.1f} 秒 |"
            )
    print("\n## 起動直後の 1 ターン目\n")
    print("| 段階 | 最初の音声まで | RTF |")
    print("|---|---:|---:|")
    for label in labels:
        d = load(label, "e2e")
        if d and "cold_first_turn" in d["data"]:
            v = d["data"]["cold_first_turn"]
            print(f"| {label} | {v['first_audio_s']:.2f} 秒 | {v['rtf']:.2f} |")


if __name__ == "__main__":
    main()
