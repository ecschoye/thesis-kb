#!/usr/bin/env python3
"""VRAM usage calculator for vLLM serving of Qwen3.5-27B and Qwen3-Embedding-8B.

Matches vLLM's actual memory allocation logic:
  available_kv = (gpu_free × gpu_memory_utilization) - model_weights - activations
  num_blocks = available_kv / block_size_bytes

Qwen3.5-27B is a hybrid model:
  - 16 full-attention layers  → standard per-token KV cache (scales with seq len)
  - 48 linear-attention layers → fixed per-sequence Mamba-style state (GatedDeltaNet)

Sources:
  - KV formula: https://lmcache.ai/kv_cache_calculator.html
  - Mamba state: vllm/model_executor/layers/mamba/mamba_utils.py
  - Memory budget: vllm/v1/worker/gpu_worker.py

Usage:
    python scripts/vram_calculator.py                           # current config
    python scripts/vram_calculator.py --max-num-seqs 96         # override
    python scripts/vram_calculator.py --preset conservative     # preset
    python scripts/vram_calculator.py --sweep                   # sweep table
    python scripts/vram_calculator.py --model qwen3-embed-8b    # embedding model
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import prod


# ── Model architectures ────────────────────────────────────────────────

@dataclass
class FullAttentionConfig:
    """Standard multi-head attention with per-token KV cache."""
    num_layers: int
    num_kv_heads: int       # GQA key-value heads
    head_dim: int           # K dimension per head
    head_dim_v: int = 0     # V dimension per head (0 = same as head_dim)

    def __post_init__(self):
        if self.head_dim_v == 0:
            self.head_dim_v = self.head_dim


@dataclass
class LinearAttentionConfig:
    """GatedDeltaNet linear attention with fixed recurrent state."""
    num_layers: int
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int


@dataclass
class ModelArch:
    name: str
    params_b: float                         # billion parameters
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    full_attn: FullAttentionConfig
    linear_attn: LinearAttentionConfig | None = None


QWEN35_27B = ModelArch(
    name="Qwen3.5-27B",
    params_b=27.0,
    hidden_size=5120,
    intermediate_size=17408,
    vocab_size=248320,
    full_attn=FullAttentionConfig(
        num_layers=16,          # every 4th of 64 layers
        num_kv_heads=4,
        head_dim=256,
    ),
    linear_attn=LinearAttentionConfig(
        num_layers=48,          # remaining 48 of 64 layers
        num_key_heads=16,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
    ),
)

QWEN3_EMBED_8B = ModelArch(
    name="Qwen3-Embedding-8B",
    params_b=8.0,
    hidden_size=4096,
    intermediate_size=12288,
    vocab_size=151665,
    full_attn=FullAttentionConfig(
        num_layers=36,
        num_kv_heads=8,
        head_dim=128,
    ),
    linear_attn=None,
)


# ── vLLM config ───────────────────────────────────────────────────────

@dataclass
class VLLMConfig:
    max_num_seqs: int = 128
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.92
    dtype_bytes: int = 2          # bf16 = 2
    tensor_parallel_size: int = 2
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    num_gpus: int = 2
    gpu_vram_gb: float = 80.0     # A100 80GB
    block_size: int = 16          # vLLM default KV cache block size (tokens)


@dataclass
class ClientConfig:
    extraction_max_workers: int = 256
    extraction_max_tokens: int = 1500
    quality_max_workers: int = 256
    quality_max_tokens: int = 2000
    quality_batch_size: int = 20
    augmentation_max_workers: int = 256
    augmentation_max_tokens: int = 2000
    unified_max_workers: int = 256


# ── Helpers ────────────────────────────────────────────────────────────

def bytes_to_gib(b: float) -> float:
    return b / (1024 ** 3)

def gib_to_bytes(gb: float) -> float:
    return gb * (1024 ** 3)

def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ── Calculator ─────────────────────────────────────────────────────────

def calculate(model: ModelArch, vllm: VLLMConfig, client: ClientConfig):
    tp = vllm.tensor_parallel_size
    db = vllm.dtype_bytes
    bs = vllm.block_size

    # ── 1. Model weights (per GPU after TP split) ─────────────────
    model_weights_bytes = model.params_b * 1e9 * db
    model_weights_per_gpu = model_weights_bytes / tp

    # ── 2. Full-attention KV cache ────────────────────────────────
    # Formula (from vLLM FullAttentionSpec.real_page_size_bytes):
    #   page_size = 2 × block_size × num_kv_heads × head_size × dtype_bytes
    # Per token (divide by block_size):
    #   kv_per_token = 2 × num_kv_heads × head_size × dtype_bytes
    # Across all full-attn layers (merged into one spec in vLLM):
    #   total = kv_per_token × num_full_layers
    # TP shards num_kv_heads across GPUs (min 1 per GPU)
    fa = model.full_attn
    kv_heads_per_gpu = max(1, fa.num_kv_heads // tp)
    full_kv_per_token_per_gpu = (
        2 * kv_heads_per_gpu * fa.head_dim * db * fa.num_layers
    )

    # Max KV for full attention (per GPU):
    #   max_num_seqs × max_model_len × per_token
    # But vLLM allocates in blocks: cdiv(max_model_len, block_size) × page_size
    full_page_size_per_gpu = 2 * bs * kv_heads_per_gpu * fa.head_dim * db * fa.num_layers
    full_kv_per_seq_per_gpu = cdiv(vllm.max_model_len, bs) * full_page_size_per_gpu
    full_kv_max_per_gpu = vllm.max_num_seqs * full_kv_per_seq_per_gpu

    # ── 3. Linear-attention state (Mamba/GatedDeltaNet) ───────────
    # Fixed per-sequence, does NOT scale with sequence length.
    # From vLLM MambaStateShapeCalculator.gated_delta_net_state_shape():
    #   conv_state: (conv_kernel_dim - 1) × (key_dim*key_heads*2 + val_dim*val_heads) / tp
    #   temporal_state: (num_v_heads / tp) × value_head_dim × key_head_dim
    # Memory per seq = (conv_elements + temporal_elements) × dtype_bytes
    # In mamba_cache_mode="none" (default for Qwen3.5):
    #   vLLM allocates 1 page per seq (line 300: page_size_bytes × 1)
    linear_state_per_seq_per_gpu = 0
    linear_state_detail = {}
    la = model.linear_attn
    if la is not None:
        conv_dim = la.key_head_dim * la.num_key_heads * 2 + la.value_head_dim * la.num_value_heads
        conv_dim_per_gpu = conv_dim // tp
        conv_elements = conv_dim_per_gpu * (la.conv_kernel_dim - 1)

        temporal_v_heads_per_gpu = la.num_value_heads // tp
        temporal_elements = temporal_v_heads_per_gpu * la.value_head_dim * la.key_head_dim

        # Per layer, per sequence
        state_per_layer = (conv_elements + temporal_elements) * db
        # Total across all linear layers
        linear_state_per_seq_per_gpu = state_per_layer * la.num_layers

        linear_state_detail = {
            "conv_dim_per_gpu": conv_dim_per_gpu,
            "conv_elements_per_layer": conv_elements,
            "temporal_elements_per_layer": temporal_elements,
            "bytes_per_layer_per_seq": state_per_layer,
            "total_bytes_per_seq_per_gpu": linear_state_per_seq_per_gpu,
        }

    linear_state_max_per_gpu = vllm.max_num_seqs * linear_state_per_seq_per_gpu

    # ── 4. Total KV + state cache ─────────────────────────────────
    total_cache_per_gpu = full_kv_max_per_gpu + linear_state_max_per_gpu

    # ── 5. Activations (transient, peak during forward pass) ──────
    # With chunked prefill, vLLM caps at max_num_batched_tokens (default 2048)
    prefill_tokens = 2048 if vllm.enable_chunked_prefill else vllm.max_model_len
    # Rough: hidden_size × ~12 (attn QKV + FFN intermediates) × dtype
    activation_per_token = model.hidden_size * 12 * db
    peak_activation = prefill_tokens * activation_per_token
    peak_activation_per_gpu = peak_activation / tp

    # ── 6. CUDA overhead ──────────────────────────────────────────
    cuda_overhead_per_gpu = gib_to_bytes(1.5)  # empirical: CUDA context + vLLM internals
    prefix_overhead_per_gpu = gib_to_bytes(0.3) if vllm.enable_prefix_caching else 0

    # ── 7. Budget ─────────────────────────────────────────────────
    usable_per_gpu = gib_to_bytes(vllm.gpu_vram_gb * vllm.gpu_memory_utilization)
    available_for_cache_per_gpu = (
        usable_per_gpu - model_weights_per_gpu
        - peak_activation_per_gpu - cuda_overhead_per_gpu
        - prefix_overhead_per_gpu
    )

    # How many sequences actually fit?
    # avg occupancy ~60% of max_model_len for full-attn KV
    for avg_fill in [0.6, 0.8, 1.0]:
        avg_full_kv_per_seq = cdiv(int(vllm.max_model_len * avg_fill), bs) * (
            2 * bs * kv_heads_per_gpu * fa.head_dim * db * fa.num_layers
        )
        cache_per_seq = avg_full_kv_per_seq + linear_state_per_seq_per_gpu
        if cache_per_seq > 0:
            if avg_fill == 0.6:
                practical_seqs_60 = int(available_for_cache_per_gpu / cache_per_seq)
            elif avg_fill == 0.8:
                practical_seqs_80 = int(available_for_cache_per_gpu / cache_per_seq)
            else:
                practical_seqs_100 = int(available_for_cache_per_gpu / cache_per_seq)

    headroom_per_gpu = available_for_cache_per_gpu - total_cache_per_gpu
    utilization_pct = ((usable_per_gpu - available_for_cache_per_gpu + total_cache_per_gpu)
                       / usable_per_gpu * 100)

    # Client pressure
    effective_workers = max(
        client.extraction_max_workers,
        client.quality_max_workers,
        client.augmentation_max_workers,
        client.unified_max_workers,
    )

    return {
        "model": model.name,
        "gpu_config": f"{vllm.num_gpus}x {vllm.gpu_vram_gb:.0f}GB (TP={tp})",
        "usable_per_gpu_gib": round(bytes_to_gib(usable_per_gpu), 2),
        "breakdown_per_gpu": {
            "model_weights": bytes_to_gib(model_weights_per_gpu),
            "full_kv_cache_max": bytes_to_gib(full_kv_max_per_gpu),
            "linear_state_max": bytes_to_gib(linear_state_max_per_gpu),
            "activations_peak": bytes_to_gib(peak_activation_per_gpu),
            "cuda_overhead": bytes_to_gib(cuda_overhead_per_gpu),
            "prefix_cache_overhead": bytes_to_gib(prefix_overhead_per_gpu),
        },
        "available_for_cache_gib": round(bytes_to_gib(available_for_cache_per_gpu), 2),
        "total_cache_requested_gib": round(bytes_to_gib(total_cache_per_gpu), 2),
        "headroom_gib": round(bytes_to_gib(headroom_per_gpu), 2),
        "utilization_pct": round(utilization_pct, 1),
        "kv_detail": {
            "full_attn_layers": fa.num_layers,
            "full_kv_per_token_per_gpu_bytes": full_kv_per_token_per_gpu,
            "full_kv_per_seq_per_gpu_mib": round(bytes_to_gib(full_kv_per_seq_per_gpu) * 1024, 2),
            "linear_attn_layers": la.num_layers if la else 0,
            "linear_state_per_seq_per_gpu_kib": round(linear_state_per_seq_per_gpu / 1024, 2) if la else 0,
            "linear_state_detail": linear_state_detail,
        },
        "concurrency": {
            "max_num_seqs": vllm.max_num_seqs,
            "practical_seqs_at_60pct": practical_seqs_60,
            "practical_seqs_at_80pct": practical_seqs_80,
            "practical_seqs_at_100pct": practical_seqs_100,
            "client_max_workers": effective_workers,
            "queue_overflow": max(0, effective_workers - vllm.max_num_seqs),
        },
        "params": {
            "max_num_seqs": vllm.max_num_seqs,
            "max_model_len": vllm.max_model_len,
            "gpu_memory_utilization": vllm.gpu_memory_utilization,
            "block_size": vllm.block_size,
            "dtype": "bfloat16" if vllm.dtype_bytes == 2 else f"{vllm.dtype_bytes*8}bit",
            "chunked_prefill": vllm.enable_chunked_prefill,
            "prefix_caching": vllm.enable_prefix_caching,
        },
    }


def print_report(result: dict):
    bd = result["breakdown_per_gpu"]
    kv = result["kv_detail"]
    cc = result["concurrency"]
    p = result["params"]

    W = 62
    print()
    print(f"{'=' * W}")
    print(f"  VRAM Calculator: {result['model']}")
    print(f"  {result['gpu_config']}")
    print(f"{'=' * W}")

    print(f"\n  vLLM Parameters:")
    print(f"  {'─' * (W-4)}")
    for k, v in p.items():
        print(f"  {k:<28} {v}")

    print(f"\n  Per-GPU Memory Budget:")
    print(f"  {'─' * (W-4)}")
    print(f"  {'Usable (gpu_vram × util):':<36} {result['usable_per_gpu_gib']:>8.2f} GiB")
    print(f"  {'─' * (W-4)}")
    print(f"  {'Model weights:':<36} {bd['model_weights']:>8.2f} GiB")
    print(f"  {'Activations (peak):':<36} {bd['activations_peak']:>8.2f} GiB")
    print(f"  {'CUDA overhead:':<36} {bd['cuda_overhead']:>8.2f} GiB")
    print(f"  {'Prefix cache metadata:':<36} {bd['prefix_cache_overhead']:>8.2f} GiB")
    non_cache = bd['model_weights'] + bd['activations_peak'] + bd['cuda_overhead'] + bd['prefix_cache_overhead']
    print(f"  {'─' * (W-4)}")
    print(f"  {'Fixed overhead:':<36} {non_cache:>8.2f} GiB")
    print(f"  {'Available for KV/state cache:':<36} {result['available_for_cache_gib']:>8.2f} GiB")
    print(f"  {'─' * (W-4)}")
    print(f"  {'Full-attn KV cache (max):':<36} {bd['full_kv_cache_max']:>8.2f} GiB")
    print(f"  {'Linear-attn state (max):':<36} {bd['linear_state_max']:>8.2f} GiB")
    total_cache = bd['full_kv_cache_max'] + bd['linear_state_max']
    print(f"  {'Total cache requested:':<36} {total_cache:>8.2f} GiB")
    print(f"  {'─' * (W-4)}")
    headroom = result['headroom_gib']
    sign = "+" if headroom >= 0 else ""
    print(f"  {'HEADROOM:':<36} {sign}{headroom:>7.2f} GiB")

    if headroom < 0:
        print(f"\n  NOTE: vLLM auto-sizes KV blocks to fit. Negative headroom means")
        print(f"  not all {p['max_num_seqs']} seqs can have {p['max_model_len']} tokens simultaneously.")
        print(f"  Sequences that exceed available blocks get preempted → timeouts.")

    print(f"\n  KV Cache Detail:")
    print(f"  {'─' * (W-4)}")
    print(f"  Full-attention:  {kv['full_attn_layers']} layers")
    print(f"    Per token/GPU: {kv['full_kv_per_token_per_gpu_bytes']:,} bytes")
    print(f"    Per seq/GPU:   {kv['full_kv_per_seq_per_gpu_mib']:.2f} MiB (at max_model_len)")
    if kv['linear_attn_layers'] > 0:
        ld = kv['linear_state_detail']
        print(f"  Linear-attention: {kv['linear_attn_layers']} layers (GatedDeltaNet)")
        print(f"    Conv state:    {ld['conv_elements_per_layer']:,} elements/layer")
        print(f"    Temporal state: {ld['temporal_elements_per_layer']:,} elements/layer")
        print(f"    Per seq/GPU:   {kv['linear_state_per_seq_per_gpu_kib']:.2f} KiB (fixed, seq-len independent)")

    print(f"\n  Concurrency Analysis:")
    print(f"  {'─' * (W-4)}")
    print(f"  vLLM max_num_seqs:         {cc['max_num_seqs']:>6}")
    print(f"  Practical fit @ 60% fill:  {cc['practical_seqs_at_60pct']:>6}")
    print(f"  Practical fit @ 80% fill:  {cc['practical_seqs_at_80pct']:>6}")
    print(f"  Practical fit @ 100% fill: {cc['practical_seqs_at_100pct']:>6}")
    print(f"  Client max_workers:        {cc['client_max_workers']:>6}")
    if cc['queue_overflow'] > 0:
        print(f"  Queue overflow:            {cc['queue_overflow']:>6} (→ timeouts)")

    # Recommendation
    safe = cc['practical_seqs_at_80pct']
    if p['max_num_seqs'] > safe:
        print(f"\n  RECOMMENDATION: Reduce max_num_seqs to ~{safe} to avoid")
        print(f"  KV pressure at typical occupancy. Also set max_workers <= {safe}.")
    elif cc['queue_overflow'] > 0:
        print(f"\n  RECOMMENDATION: Reduce max_workers from {cc['client_max_workers']} to")
        print(f"  {p['max_num_seqs']} (or {p['max_num_seqs'] + 32} for pipelining).")

    print(f"\n{'=' * W}")


def sweep(model: ModelArch, vllm_cfg: VLLMConfig, client: ClientConfig):
    W = 80
    print(f"\n  {'─' * W}")
    print(f"  Sweep: max_num_seqs (max_model_len={vllm_cfg.max_model_len}, "
          f"util={vllm_cfg.gpu_memory_utilization}, TP={vllm_cfg.tensor_parallel_size})")
    print(f"  {'─' * W}")
    print(f"  {'seqs':>6}  {'FullKV/GPU':>10}  {'LinSt/GPU':>10}  {'Cache':>8}  "
          f"{'Avail':>8}  {'Headroom':>9}  {'Fit@80%':>8}  {'Risk':>6}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*6}")

    for seqs in [48, 64, 80, 96, 112, 128, 160, 192, 224, 256]:
        v = VLLMConfig(
            max_num_seqs=seqs,
            max_model_len=vllm_cfg.max_model_len,
            gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
            dtype_bytes=vllm_cfg.dtype_bytes,
            tensor_parallel_size=vllm_cfg.tensor_parallel_size,
            enable_chunked_prefill=vllm_cfg.enable_chunked_prefill,
            enable_prefix_caching=vllm_cfg.enable_prefix_caching,
            num_gpus=vllm_cfg.num_gpus,
            gpu_vram_gb=vllm_cfg.gpu_vram_gb,
            block_size=vllm_cfg.block_size,
        )
        c = ClientConfig(unified_max_workers=seqs,
                         extraction_max_workers=seqs,
                         quality_max_workers=seqs,
                         augmentation_max_workers=seqs)
        r = calculate(model, v, c)
        bd = r["breakdown_per_gpu"]
        cc = r["concurrency"]
        fit80 = cc["practical_seqs_at_80pct"]

        if seqs <= fit80:
            risk = "LOW"
        elif seqs <= fit80 * 1.3:
            risk = "MED"
        else:
            risk = "HIGH"

        marker = " <--" if seqs == vllm_cfg.max_num_seqs else ""
        print(f"  {seqs:>6}  {bd['full_kv_cache_max']:>9.2f}G  {bd['linear_state_max']:>9.2f}G  "
              f"{r['total_cache_requested_gib']:>7.2f}G  {r['available_for_cache_gib']:>7.2f}G  "
              f"{r['headroom_gib']:>+8.2f}G  {fit80:>7}  {risk:>6}{marker}")

    print()


def main():
    parser = argparse.ArgumentParser(description="VRAM calculator for vLLM")
    parser.add_argument("--model", choices=["qwen35-27b", "qwen3-embed-8b"], default="qwen35-27b")
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--gpu-vram-gb", type=float, default=80.0)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--max-workers", type=int, help="Override all client max_workers")
    parser.add_argument("--quality-batch-size", type=int)
    parser.add_argument("--no-chunked-prefill", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Sweep table")
    parser.add_argument("--preset", choices=["current", "conservative", "balanced"])
    args = parser.parse_args()

    if args.model == "qwen3-embed-8b":
        model = QWEN3_EMBED_8B
        defaults = dict(max_num_seqs=256, max_model_len=2048,
                        gpu_memory_utilization=0.92, tp=2, block_size=16)
    else:
        model = QWEN35_27B
        defaults = dict(max_num_seqs=128, max_model_len=8192,
                        gpu_memory_utilization=0.92, tp=2, block_size=16)

    if args.preset == "conservative":
        defaults.update(max_num_seqs=96, gpu_memory_utilization=0.88)
    elif args.preset == "balanced":
        defaults.update(max_num_seqs=128, gpu_memory_utilization=0.90)

    vllm_cfg = VLLMConfig(
        max_num_seqs=args.max_num_seqs or defaults["max_num_seqs"],
        max_model_len=args.max_model_len or defaults["max_model_len"],
        gpu_memory_utilization=args.gpu_memory_utilization or defaults["gpu_memory_utilization"],
        tensor_parallel_size=args.tensor_parallel_size or defaults["tp"],
        enable_chunked_prefill=not args.no_chunked_prefill,
        enable_prefix_caching=not args.no_prefix_caching,
        num_gpus=args.num_gpus or defaults["tp"],
        gpu_vram_gb=args.gpu_vram_gb,
        block_size=args.block_size or defaults["block_size"],
    )

    workers = args.max_workers or vllm_cfg.max_num_seqs
    client_cfg = ClientConfig(
        extraction_max_workers=workers,
        quality_max_workers=workers,
        augmentation_max_workers=workers,
        unified_max_workers=workers,
        quality_batch_size=args.quality_batch_size or 20,
    )

    result = calculate(model, vllm_cfg, client_cfg)
    print_report(result)

    if args.sweep:
        sweep(model, vllm_cfg, client_cfg)


if __name__ == "__main__":
    main()
