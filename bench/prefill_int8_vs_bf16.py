import sys, time, torch
from pathlib import Path
sys.path.insert(0, "/Users/yuhi/workspace/LFM-Audio-Test")
from huggingface_hub import snapshot_download
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from accel import quantize_int8
from filler import make_filler
d = Path(snapshot_download("LiquidAI/LFM2.5-Audio-1.5B-JP", ignore_patterns=["demo.mp4"]))
proc = LFM2AudioProcessor.from_pretrained(d, device="mps").eval()
model = LFM2AudioModel.from_pretrained(d, device="cpu").eval()
if sys.argv[1] == "int8": quantize_int8(model)
model.to("mps")
count = lambda s: len(proc.text.encode(s, add_special_tokens=False))
chat = ChatState(proc); chat.new_turn("system"); chat.add_text("Respond with interleaved text and audio." + make_filler(8000, count)); chat.end_turn()
ts = []
with torch.no_grad():
    for _ in range(4):
        torch.mps.synchronize(); t = time.perf_counter()
        model.lfm(inputs_embeds=model._prefill(**chat), use_cache=True); torch.mps.synchronize()
        ts.append(round(time.perf_counter() - t, 2))
import psutil
print(f"{sys.argv[1]:5} 8k prefill: {ts} (swap {psutil.swap_memory().used / 1e9:.1f}GB)")
