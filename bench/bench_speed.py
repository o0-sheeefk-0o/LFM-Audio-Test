"""速度計測

  uv run python bench/bench_speed.py components --opts none --label baseline   # 1 ステップの内訳
  uv run python bench/bench_speed.py e2e --opts none --label baseline          # 1 ターン全体
  uv run python bench/bench_speed.py prefill --opts none --label baseline      # system の長さと prefill 時間
  uv run python bench/bench_speed.py multiturn --opts all --label opt4         # 長い system + 複数ターンの遅延

--opts は有効にする最適化（accel.Optimizations のフィールド名をカンマ区切り / all / none）。
結果は bench/results/<label>-<mode>.json に保存する。

components の各項目は 3 種類の時間を出す:
  latency_ms   1 回ごとに GPU 同期して測った時間
  cpu_ms       Python から GPU にカーネルを積み終わるまでの時間（同期なし）
  pipelined_ms 同期せずに連続実行したときの 1 回あたりの時間
cpu_ms ≈ pipelined_ms なら CPU（Python / カーネル起動）律速、cpu_ms ≪ pipelined_ms なら GPU 律速。
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import soundfile as sf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accel import Optimizations  # noqa: E402
from engine import BASE_INSTRUCTION, Engine, GenSettings  # noqa: E402
from filler import make_filler  # noqa: E402
from liquid_audio import ChatState  # noqa: E402
from tools import SAMPLE_TOOLS_JSON  # noqa: E402

FIXTURES = ROOT / "bench" / "fixtures"
RESULTS = ROOT / "bench" / "results"
CUSTOM = "あなたは親切な日本語の音声アシスタントです。回答は短く、話し言葉で答えてください。"


def load_fixture(name: str) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(FIXTURES / f"{name}.wav", dtype="float32")
    return torch.from_numpy(wav)[None], sr


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def seed(n: int = 0) -> None:
    torch.manual_seed(n)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(n)


def timeit(fn: Callable[[], object], device: torch.device, n: int = 30, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    sync(device)

    latency = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync(device)
        latency.append(time.perf_counter() - t0)

    cpu = []
    start = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        cpu.append(time.perf_counter() - t0)
    sync(device)
    pipelined = (time.perf_counter() - start) / n

    return {
        "latency_ms": statistics.median(latency) * 1e3,
        "cpu_ms": statistics.median(cpu) * 1e3,
        "pipelined_ms": pipelined * 1e3,
    }


def prefill_chat(engine: Engine, system: str, fixture: str) -> ChatState:
    chat = ChatState(engine.proc)
    chat.new_turn("system")
    chat.add_text(system)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(*load_fixture(fixture))
    chat.end_turn()
    chat.new_turn("assistant")
    return chat


@torch.no_grad()
def bench_components(engine: Engine) -> dict[str, dict[str, float]]:
    model, device = engine.model, engine.device
    chat = prefill_chat(engine, BASE_INSTRUCTION + "\n" + CUSTOM, "weather")
    out = model.lfm(inputs_embeds=model._prefill(**chat), use_cache=True)
    cache = out.past_key_values
    h = out.last_hidden_state[0, -1]
    x = h[None, None, :].clone()
    codes = torch.randint(0, 2048, (8,), device=device)

    text_head = engine.text_head or (lambda v: nn.functional.linear(v, model.lfm.embed_tokens.weight))
    sample_frame = engine.depthformer.sample if engine.depthformer else model._sample_audio_frame

    results: dict[str, dict[str, float]] = {}
    results["lfm_step"] = timeit(lambda: model.lfm(inputs_embeds=x, past_key_values=cache, use_cache=True), device)
    results["text_head"] = timeit(lambda: model._sample_text_token(text_head(h)), device)
    results["audio_frame"] = timeit(lambda: sample_frame(h, temperature=1.0, top_k=4), device)
    results["audio_embed"] = timeit(lambda: model.audio_embedding(codes + model.codebook_offsets).sum(0), device)
    mimi_device = next(engine.mimi.parameters()).device
    mimi_codes = codes.to(mimi_device)[None, :, None]
    with engine.mimi.streaming(1):
        results["mimi_decode"] = timeit(lambda: engine.mimi.decode(mimi_codes), device)
    results["to_cpu_sync"] = timeit(lambda: codes.cpu(), device)

    text = results["lfm_step"]["latency_ms"] + results["text_head"]["latency_ms"]
    # Mimi を CPU の別スレッドで動かす場合は GPU と並列なので、音声ステップの律速経路から外す
    audio_keys = ["lfm_step", "audio_frame", "audio_embed"] + ([] if engine.opts.mimi_on_cpu else ["mimi_decode"])
    audio = sum(results[k]["latency_ms"] for k in audio_keys)
    # interleaved は テキスト 6 : 音声 9。音声 1 フレーム = 80ms
    per_cycle = 6 * text + 9 * audio
    results["estimate"] = {"text_step_ms": text, "audio_step_ms": audio, "rtf": per_cycle / (9 * 80)}
    return results


def bench_e2e(engine: Engine, repeats: int = 3) -> dict[str, object]:
    cases = {
        "tool_weather": ("weather", GenSettings(system_prompt=CUSTOM, tools_enabled=True, tools_json=SAMPLE_TOOLS_JSON)),
        "no_tool_hobby": ("hobby", GenSettings(system_prompt=CUSTOM, tools_enabled=True, tools_json=SAMPLE_TOOLS_JSON)),
        "tools_off_fuji": ("fuji", GenSettings(system_prompt=CUSTOM, tools_enabled=False)),
    }
    out: dict[str, object] = {}
    cold_done = False
    for name, (fixture, settings) in cases.items():
        runs = []
        for i in range(repeats + (0 if cold_done else 1)):
            seed(i)
            session = engine.new_session()
            start = time.perf_counter()
            first_audio = None
            for kind, _ in engine.run_user_turn(session, *load_fixture(fixture), settings):
                if kind == "audio" and first_audio is None:
                    first_audio = time.perf_counter() - start
            total = time.perf_counter() - start
            answer = [s for s in session.stats if s.kind == "応答"][-1]
            route = [s for s in session.stats if s.kind == "ツール判定"]
            run = {
                "first_audio_s": first_audio,
                "total_s": total,
                "route_s": route[0].elapsed if route else 0.0,
                "answer_ttft_s": answer.ttft,
                "steps_per_s": answer.steps / answer.elapsed,
                "audio_s": answer.audio_seconds,
                "rtf": answer.elapsed / answer.audio_seconds if answer.audio_seconds else None,
                "text": session.messages[-1]["content"][:60],
            }
            if not cold_done:
                out["cold_first_turn"] = {"case": name, **run}
                cold_done = True
                continue
            runs.append(run)
        keys = ["first_audio_s", "total_s", "route_s", "answer_ttft_s", "steps_per_s", "rtf"]
        out[name] = {k: statistics.median(r[k] for r in runs) for k in keys} | {"runs": runs}
    return out


def bench_multiturn(engine: Engine) -> list[dict[str, object]]:
    """system の長さ × 3 ターン会話で、ターンごとの「最初の音声まで」を測る（prefix キャッシュの効果を見る）"""
    rows = []
    for size in [0, 2000, 8000]:
        system = CUSTOM + (make_filler(size, engine.count_tokens) if size else "")
        settings = GenSettings(system_prompt=system, tools_enabled=True, tools_json=SAMPLE_TOOLS_JSON, max_new_tokens=400)
        session = engine.new_session()
        for turn, fixture in enumerate(["weather", "fuji", "hobby"], 1):
            seed(turn)
            n_stats = len(session.stats)
            start = time.perf_counter()
            first_audio = None
            for kind, _ in engine.run_user_turn(session, *load_fixture(fixture), settings):
                if kind == "audio" and first_audio is None:
                    first_audio = time.perf_counter() - start
            stats = session.stats[n_stats:]
            route = [s for s in stats if s.kind == "ツール判定"]
            answer = [s for s in stats if s.kind == "応答"][-1]
            rows.append(
                {
                    "filler_tokens": size,
                    "turn": turn,
                    "first_audio_s": first_audio,
                    "route_s": route[0].elapsed if route else 0.0,
                    "route_prefill": route[0].prefill_tokens if route else 0,
                    "answer_ttft_s": answer.ttft,
                    "answer_prefill": answer.prefill_tokens,
                    "context_tokens": answer.context_tokens,
                    "text": session.messages[-1]["content"][:50],
                }
            )
            print(rows[-1], flush=True)
    return rows


@torch.no_grad()
def bench_prefill(engine: Engine) -> list[dict[str, float]]:
    model, device = engine.model, engine.device
    rows = []
    for size in [0, 1000, 2000, 4000, 8000]:
        system = BASE_INSTRUCTION + "\n" + CUSTOM + (make_filler(size, engine.count_tokens) if size else "")
        chat = prefill_chat(engine, system, "fuji")

        def run():
            emb = model._prefill(**chat)
            model.lfm(inputs_embeds=emb, use_cache=True)

        r = timeit(run, device, n=3, warmup=1)
        rows.append({"filler_tokens": size, "context_tokens": int(chat.modality_flag.shape[1]), "prefill_ms": r["latency_ms"]})
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["components", "e2e", "prefill", "multiturn"])
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--opts", default="all", help="有効にする最適化（カンマ区切り / all / none）")
    args = parser.parse_args()

    os.environ["LFM_OPTS"] = args.opts
    load_start = time.perf_counter()
    engine = Engine(opts=Optimizations.from_env())
    load_s = time.perf_counter() - load_start

    result = {"label": args.label, "mode": args.mode, "opts": engine.opts.label(), "device": str(engine.device)}
    result["env"] = {"torch": torch.__version__, "machine": platform.machine(), "python": platform.python_version(), "load_s": load_s}
    modes = {"components": bench_components, "e2e": bench_e2e, "prefill": bench_prefill, "multiturn": bench_multiturn}
    result["data"] = modes[args.mode](engine)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{args.label}-{args.mode}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["data"], ensure_ascii=False, indent=2, default=str)[:4000])
    print(f"saved {path}")


if __name__ == "__main__":
    main()
