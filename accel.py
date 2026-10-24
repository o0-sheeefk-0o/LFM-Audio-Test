"""推論の高速化部品

各最適化は Optimizations で個別にオン/オフできる（環境変数 LFM_OPTS でも指定可）。
計測結果と各最適化の効果は docs/performance.md を参照。
"""

import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from liquid_audio import LFM2AudioModel, moshi

AUDIO_EOS = 2048


@dataclass
class Optimizations:
    fast_depthformer: bool = True  # 音声フレーム生成（depthformer）を書き直して torch.compile
    mimi_on_cpu: bool = True  # Mimi デコードを CPU の別スレッドで行い、GPU の律速経路から外す
    int8: bool = True  # Linear の重みを int8 にして、メモリ帯域律速の decode を速くする
    prefix_cache: bool = True  # system プロンプトと会話履歴の KV キャッシュをターンをまたいで使い回す

    @classmethod
    def from_env(cls) -> "Optimizations":
        """LFM_OPTS="all"（既定）/ "none" / "fast_depthformer,mimi_on_cpu" のように指定する"""
        spec = os.environ.get("LFM_OPTS", "all").strip()
        if spec == "all":
            return cls()
        enabled = {s.strip() for s in spec.split(",") if s.strip() and s.strip() != "none"}
        unknown = enabled - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown optimization(s): {unknown}")
        return cls(**{f.name: f.name in enabled for f in fields(cls)})

    def label(self) -> str:
        on = [f.name for f in fields(self) if getattr(self, f.name)]
        return ",".join(on) or "none"


# ---------------------------------------------------------------------------
# int8 重み量子化
# ---------------------------------------------------------------------------


class Int8Linear(nn.Module):
    """重みを出力チャネルごとのスケール付き int8 で持つ Linear

    decode（入力 1 行）は重みの読み出しが律速なので、torch._weight_int8pack_mm で読み出し量を半分にする。
    入力が多数行（prefill）でも MPS の _weight_int8pack_mm は bf16 の行列積と同等の速さなので常にこれを使う
    （bf16 に戻してから計算すると、戻す処理だけで 1 回 5ms 以上かかる）。
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        w = weight.detach().float()
        scales = w.abs().amax(1).clamp_min(1e-8) / 127
        self.register_buffer("weight_int8", torch.round(w / scales[:, None]).clamp(-128, 127).to(torch.int8))
        self.register_buffer("scales", scales.to(weight.dtype))
        self.bias = None if bias is None else nn.Parameter(bias.detach(), requires_grad=False)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        return cls(linear.weight, linear.bias)

    @classmethod
    def concat(cls, a: "Int8Linear", b: "Int8Linear") -> "Int8Linear":
        """同じ入力にかかる 2 つの Linear を出力方向に連結して 1 回の行列積にする"""
        fused = cls.__new__(cls)
        nn.Module.__init__(fused)
        fused.register_buffer("weight_int8", torch.cat([a.weight_int8, b.weight_int8]))
        fused.register_buffer("scales", torch.cat([a.scales, b.scales]))
        fused.bias = None
        fused.in_features = a.in_features
        fused.out_features = a.out_features + b.out_features
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch._weight_int8pack_mm(x.reshape(-1, x.shape[-1]), self.weight_int8, self.scales).view(*x.shape[:-1], -1)
        if self.bias is not None:
            y = y + self.bias
        return y


def quantize_int8(model: LFM2AudioModel) -> Callable[[torch.Tensor], torch.Tensor]:
    """LFM2 バックボーンと depthformer の Linear を Int8Linear に置き換え、int8 のテキストヘッドを返す

    音声エンコーダ（conformer）は prefill でしか使わないので対象外。
    _weight_int8pack_mm は入出力の次元が 32 の倍数でないと使えないので、
    depthformer の出力層（語彙 2049）はそのまま残す（200 万パラメータ程度で影響は小さい）。
    """
    targets = [model.lfm.layers, model.depthformer, model.depth_embeddings]
    for root in targets:
        for module in list(root.modules()):
            for child_name, child in list(module.named_children()):
                if isinstance(child, nn.Linear) and child.in_features % 32 == 0 and child.out_features % 32 == 0:
                    setattr(module, child_name, Int8Linear.from_linear(child))
    model.depth_linear = Int8Linear.from_linear(model.depth_linear)

    # テキストヘッドは埋め込み行列と重み共有なので、ロジット計算用に int8 の複製を別に持つ。
    # model の子モジュールにして、model.to() で一緒にデバイスへ移るようにする
    model.text_head_int8 = Int8Linear(model.lfm.embed_tokens.weight, None)
    return lambda h: model.text_head_int8(h[None])[0]


# ---------------------------------------------------------------------------
# depthformer（音声フレームのサンプリング）
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # liquid_audio.model.transformer.RMSNorm と同じく float32 で正規化してから元の dtype に戻す
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def _concat_linear(a: nn.Module, b: nn.Module) -> nn.Module:
    if isinstance(a, Int8Linear) and isinstance(b, Int8Linear):
        return Int8Linear.concat(a, b)
    assert isinstance(a, nn.Linear) and isinstance(b, nn.Linear) and a.bias is None and b.bias is None
    fused = nn.Linear(a.in_features, a.out_features + b.out_features, bias=False, device=a.weight.device, dtype=a.weight.dtype)
    fused.weight.data = torch.cat([a.weight.data, b.weight.data])
    return fused


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # 隣り合う 2 要素を複素数とみなして回転する（元実装の view_as_complex 版と同じ並び）
    xf = x.float().unflatten(-1, (-1, 2))
    a, b = xf[..., 0], xf[..., 1]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2).to(x.dtype)


class FastDepthformer:
    """LFM2AudioModel._sample_audio_frame（音声 1 フレーム = 8 コードブックの逐次サンプリング）の高速版

    元実装は 1 フレームあたり 8 コードブック × 6 層を、複素数の RoPE・torch.cat の KV キャッシュ・
    topk + masked_fill + softmax + multinomial のサンプリングで回しており、小さいカーネルの起動が支配的だった。
    ここでは次のように書き直して torch.compile で 1 つのグラフにまとめる。
      - 位置は常に 0〜7 なので RoPE の cos/sin を事前計算し、実数演算にする
      - 8 コードブックのループを展開して形状を静的にする
      - top-k サンプリングを Gumbel-max（topk した値にノイズを足して argmax）にする。分布は同じ
      - temperature はテンソルで渡し、値を変えても再コンパイルしない（top_k は変えると再コンパイル）
      - 長さ 8 以下の注意機構は bmm ではなく要素積 + 和で書き、周辺の演算と 1 カーネルに融合させる
      - GLU の w1 と w3 を連結して 1 回の行列積にする
    """

    def __init__(self, model: LFM2AudioModel, *, compile: bool = True) -> None:
        self.model = model
        self.codebooks = model.codebooks
        mha = model.depthformer.layers[0].operator
        attn = mha.bounded_attention
        self.dim = mha.dim
        self.n_heads = mha.num_heads
        self.n_kv = mha.gqa_dim
        self.head_dim = mha.head_dim
        self.scale = 1 / math.sqrt(self.head_dim)
        self.norm_eps = attn.q_layernorm.eps

        device = model.codebook_offsets.device
        freqs = mha.freqs_cis[: self.codebooks].to(device)
        self.cos = freqs.real.contiguous()
        self.sin = freqs.imag.contiguous()
        self._temperatures: dict[float, torch.Tensor] = {}
        self.w13 = [_concat_linear(layer.feed_forward.w1, layer.feed_forward.w3) for layer in model.depthformer.layers]
        self.ff_dim = model.depthformer.layers[0].feed_forward.w1.out_features

        self._frame =torch.compile(self._frame_impl, dynamic=False) if compile else self._frame_impl

    @torch.no_grad()
    def sample(self, embedding: torch.Tensor, *, temperature: float | None, top_k: int | None) -> torch.Tensor:
        if temperature is None or temperature <= 0 or top_k == 1:
            return self._frame(embedding, None, None, None)
        n = top_k or self.model.audio_vocab_size
        # Gumbel-max 用の一様乱数。コンパイル済みグラフの外で作る
        noise = torch.rand((self.codebooks, n), device=embedding.device)
        return self._frame(embedding, noise, self._temperature_tensor(float(temperature), embedding.device), top_k)

    def _temperature_tensor(self, value: float, device: torch.device) -> torch.Tensor:
        # CPU から MPS への転送は GPU の処理待ち（同期）になるので、毎フレーム作らずに使い回す
        if value not in self._temperatures:
            self._temperatures[value] = torch.tensor(value, device=device)
        return self._temperatures[value]

    def _layer(self, idx: int, layer: nn.Module, x: torch.Tensor, pos: int, keys: list[torch.Tensor], values: list[torch.Tensor]) -> torch.Tensor:
        mha = layer.operator
        attn = mha.bounded_attention

        h = _rms_norm(x, layer.operator_norm.weight, layer.operator_norm.eps)
        q, k, v = mha.qkv_proj(h).split([self.dim, self.head_dim * self.n_kv, self.head_dim * self.n_kv])
        q = _rms_norm(q.view(self.n_heads, self.head_dim), attn.q_layernorm.weight, self.norm_eps)
        k = _rms_norm(k.view(self.n_kv, self.head_dim), attn.k_layernorm.weight, self.norm_eps)
        q = _rope(q, self.cos[pos], self.sin[pos])
        k = _rope(k, self.cos[pos], self.sin[pos])
        keys.append(k)
        values.append(v.view(self.n_kv, self.head_dim))

        # GQA: クエリヘッド h は KV ヘッド h // (n_heads / n_kv) を見る
        K = torch.stack(keys, 1).float()  # [n_kv, L, head_dim]
        V = torch.stack(values, 1).float()
        qg = q.view(self.n_kv, self.n_heads // self.n_kv, 1, self.head_dim).float()
        scores = (qg * K[:, None]).sum(-1) * self.scale  # [n_kv, group, L]
        o = (scores.softmax(-1)[..., None] * V[:, None]).sum(2).to(x.dtype).reshape(self.dim)
        h = x + mha.out_proj(o)

        f = _rms_norm(h, layer.ffn_norm.weight, layer.ffn_norm.eps)
        g, u = self.w13[idx](f).split(self.ff_dim)
        return h + layer.feed_forward.w2(F.silu(g) * u)

    def _frame_impl(self, embedding: torch.Tensor, noise: torch.Tensor | None, temperature: torch.Tensor | None, top_k: int | None) -> torch.Tensor:
        model = self.model
        depth_in = model.depth_linear(embedding).view(self.codebooks, self.dim)
        prev = torch.zeros_like(depth_in[0])
        caches = [([], []) for _ in model.depthformer.layers]

        tokens = []
        for i in range(self.codebooks):
            x = depth_in[i] + prev
            for idx, (layer, (keys, values)) in enumerate(zip(model.depthformer.layers, caches, strict=True)):
                x = self._layer(idx, layer, x, i, keys, values)

            emb = model.depth_embeddings[i]
            logits = emb.to_logits(_rms_norm(x, emb.embedding_norm.weight, emb.embedding_norm.eps)).float()
            if noise is None:
                token = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                values_k, index_k = logits.topk(top_k) if top_k is not None else (logits, None)
                gumbel = -torch.log(-torch.log(noise[i].clamp_min(1e-20)))
                choice = (values_k + gumbel).argmax(-1, keepdim=True)
                token = index_k[choice] if index_k is not None else choice
            tokens.append(token)
            prev = F.embedding(token, emb.embedding.weight)[0]

        return torch.cat(tokens)


# ---------------------------------------------------------------------------
# Mimi デコード
# ---------------------------------------------------------------------------


def load_cpu_mimi(mimi_weights_path: str | Path) -> Any:
    from safetensors.torch import load_file

    mimi = moshi.models.loaders.get_mimi(None, device="cpu")
    mimi.load_state_dict(load_file(str(mimi_weights_path), device="cpu"), strict=True)
    return mimi.eval()


class MimiWorker:
    """音声コードを受け取って CPU の別スレッドで Mimi デコードし、波形を out キューに積む

    GPU（MPS）で生成しながら CPU でデコードするので、デコード時間が生成の律速経路から外れる。
    たまっているフレームはまとめてデコードする（1 フレームずつより速い）。
    """

    MAX_BATCH = 4

    def __init__(self, mimi: Any, out: Queue) -> None:
        self.mimi = mimi
        self.out = out
        self.frames: Queue[torch.Tensor | None] = Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def put(self, codes: torch.Tensor) -> None:
        """codes: CPU 上の (8,) の音声コード。EOS フレームは渡さない"""
        self.frames.put(codes)

    def close(self) -> None:
        self.frames.put(None)
        self.thread.join()

    @torch.no_grad()
    def _run(self) -> None:
        done = False
        try:
            with self.mimi.streaming(1):
                while not done:
                    batch = [self.frames.get()]
                    while len(batch) < self.MAX_BATCH:
                        try:
                            batch.append(self.frames.get_nowait())
                        except Empty:
                            break
                    if batch[-1] is None:
                        batch.pop()
                        done = True
                    if batch:
                        codes = torch.stack(batch, 1)[None]  # (1, 8, T)
                        wav = self.mimi.decode(codes)[0, 0]
                        self.out.put(("wav", wav.numpy().astype(np.float32)))
        except Exception as e:  # noqa: BLE001 - 呼び出し側で再送出する
            self.out.put(("error", e))
