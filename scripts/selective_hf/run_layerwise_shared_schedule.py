"""Run HF eval on an EMO checkpoint with a per-layer num_shared_experts schedule.

Materializes a working directory next to the HF cache (symlinks for weights +
tokenizer, fresh config.json with ``num_shared_experts_per_layer`` set), optionally
verifies the override propagated end-to-end, then dispatches
``src.scripts.eval.launch_eval``.

Example:
    python -m scripts.selective_hf.run_layerwise_shared_schedule \\
        --model allenai/Emo_1b14b_1T \\
        --schedule up_linear \\
        --task arc_challenge \\
        --output-dir claude_outputs/shared_schedule_eval
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# -------- schedule construction --------

def _monotonic_int_schedule(L: int, target_sum: int, low: int, high: int) -> List[int]:
    """Non-decreasing integer schedule of length L that sums to ``target_sum``.

    Starts from a linear interpolation low->high (rounded), then nudges
    individual entries until the sum matches while preserving non-decreasing
    order and the [low, high] bounds.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if not (low <= high):
        raise ValueError("low must be <= high")
    if target_sum < low * L or target_sum > high * L:
        raise ValueError(
            f"target_sum={target_sum} infeasible for L={L}, low={low}, high={high}"
        )

    sched = [round(low + (high - low) * i / max(L - 1, 1)) for i in range(L)]
    diff = target_sum - sum(sched)

    # Nudge while preserving non-decreasing order.
    # If sum is short, try to +1 the rightmost position that won't exceed `high`
    # and remains <= its right neighbor (trivially true at the end).
    # If sum is over, mirror.
    while diff != 0:
        if diff > 0:
            for i in range(L - 1, -1, -1):
                if sched[i] >= high:
                    continue
                if i + 1 < L and sched[i] + 1 > sched[i + 1]:
                    continue
                sched[i] += 1
                diff -= 1
                break
            else:
                raise RuntimeError("could not increase schedule further")
        else:
            for i in range(L):
                if sched[i] <= low:
                    continue
                if i > 0 and sched[i] - 1 < sched[i - 1]:
                    continue
                sched[i] -= 1
                diff += 1
                break
            else:
                raise RuntimeError("could not decrease schedule further")

    assert sum(sched) == target_sum
    assert all(sched[i] <= sched[i + 1] for i in range(L - 1)), "not non-decreasing"
    return sched


def resolve_schedule(name: str, L: int, K_base: int, top_k: int) -> List[int]:
    """Resolve a preset name (or comma-separated explicit list) to a schedule."""
    target_sum = L * K_base

    if "," in name or name.startswith("["):
        cleaned = name.strip().lstrip("[").rstrip("]")
        sched = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
        if len(sched) != L:
            raise ValueError(f"explicit schedule length {len(sched)} != L={L}")
        return sched

    if name == "uniform":
        return [K_base] * L

    if name == "up_step":
        # Half zeros, half 2*K_base. Sum = 0*(L/2) + 2*K_base*(L/2) = K_base*L.
        if L % 2 != 0:
            raise ValueError("up_step requires even L")
        return [0] * (L // 2) + [2 * K_base] * (L // 2)

    if name == "down_step":
        if L % 2 != 0:
            raise ValueError("down_step requires even L")
        return [2 * K_base] * (L // 2) + [0] * (L // 2)

    if name == "up_linear":
        high = min(2 * K_base + 1, top_k)
        return _monotonic_int_schedule(L, target_sum, low=0, high=high)

    if name == "down_linear":
        high = min(2 * K_base + 1, top_k)
        sched = _monotonic_int_schedule(L, target_sum, low=0, high=high)
        return sched[::-1]

    raise ValueError(f"unknown schedule preset: {name}")


# -------- working dir materialization --------

def _resolve_snapshot_dir(model_path_or_id: str) -> Path:
    """Return a local snapshot directory containing the HF checkpoint files."""
    p = Path(model_path_or_id)
    if p.exists() and p.is_dir():
        return p.resolve()
    # Treat as HF hub repo id, download a fully materialized snapshot.
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path_or_id)).resolve()


def materialize_work_dir(snapshot_dir: Path, work_dir: Path, schedule: List[int]) -> Path:
    """Symlink everything from ``snapshot_dir`` into ``work_dir`` except config.json,
    then write a new config.json with ``num_shared_experts_per_layer`` set.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    for entry in snapshot_dir.iterdir():
        target = work_dir / entry.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if entry.name == "config.json":
            continue
        # Resolve through any symlinks in the snapshot dir so we point at real
        # blob files rather than chained symlinks.
        target.symlink_to(entry.resolve())

    with open(snapshot_dir / "config.json") as f:
        cfg = json.load(f)
    cfg["num_shared_experts_per_layer"] = schedule
    with open(work_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return work_dir


# -------- smoke test --------

def smoke_test(work_dir: Path, schedule: List[int]) -> None:
    """Load model from work_dir and assert each layer's MoE block received the
    intended per-layer ``num_shared_experts`` value."""
    print(f"[smoke] loading model from {work_dir} ...", flush=True)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(work_dir), trust_remote_code=True)
    assert list(config.num_shared_experts_per_layer) == list(schedule), (
        f"config did not absorb override: "
        f"{config.num_shared_experts_per_layer} vs {schedule}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(work_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    layers = model.model.layers if hasattr(model, "model") else model.layers
    seen = []
    for i, layer in enumerate(layers):
        # The MoE block lives at layer.mlp (EmoSparseMoeBlock) per modeling_emo.py
        moe = layer.mlp
        seen.append(int(moe.num_shared_experts))
    print(f"[smoke] observed per-layer num_shared_experts: {seen}")
    assert seen == schedule, f"per-layer override not honored: got {seen}, want {schedule}"
    print("[smoke] OK")
    del model


# -------- main --------

def _read_config_baseline(model_path_or_id: str) -> Tuple[int, int, int, Path]:
    snap = _resolve_snapshot_dir(model_path_or_id)
    with open(snap / "config.json") as f:
        cfg = json.load(f)
    L = int(cfg["num_hidden_layers"])
    K_base = int(cfg.get("num_shared_experts", 0))
    top_k = int(cfg.get("num_experts_per_tok", 0))
    return L, K_base, top_k, snap


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF hub repo id or local snapshot path (e.g. allenai/Emo_1b14b_1T)")
    p.add_argument("--schedule", required=True,
                   help="Preset name (uniform, up_linear, down_linear, up_step, down_step) "
                        "or comma-separated explicit list of length num_hidden_layers")
    p.add_argument("--task", action="append", default=[],
                   help="Eval task; pass multiple times for multiple tasks")
    p.add_argument("--output-dir", required=True,
                   help="Root output dir; per-schedule subdir will be created")
    p.add_argument("--work-root", default="claude_outputs/shared_schedule_eval/work",
                   help="Where to materialize overridden snapshot dirs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num-shots", type=int, default=None)
    p.add_argument("--limit", type=float, default=None,
                   help="Subsample fraction or count, forwarded to launch_eval")
    p.add_argument("--smoke-test", action="store_true",
                   help="Load model on CPU and verify per-layer override propagated before eval")
    p.add_argument("--no-eval", action="store_true",
                   help="Stop after materializing the work dir (and smoke test if enabled)")
    p.add_argument("--print-schedule-only", action="store_true",
                   help="Resolve and print the schedule, then exit (no downloads beyond config.json)")
    args = p.parse_args(argv)

    L, K_base, top_k, snap = _read_config_baseline(args.model)
    print(f"[info] model baseline: L={L}, num_shared_experts={K_base}, top_k={top_k}")

    schedule = resolve_schedule(args.schedule, L, K_base, top_k)
    print(f"[info] schedule '{args.schedule}': {schedule}  (sum={sum(schedule)}, baseline sum={L*K_base})")

    if args.print_schedule_only:
        return 0

    work_dir = Path(args.work_root) / args.schedule.replace(",", "_").replace("[", "").replace("]", "")
    materialize_work_dir(snap, work_dir, schedule)
    print(f"[info] materialized work dir at {work_dir}")
    # Record the resolved schedule and provenance for the run.
    with open(work_dir / "schedule_meta.json", "w") as f:
        json.dump({
            "model": args.model,
            "snapshot_dir": str(snap),
            "schedule_name": args.schedule,
            "schedule": schedule,
            "L": L,
            "K_base": K_base,
            "top_k": top_k,
            "target_sum": L * K_base,
        }, f, indent=2)

    if args.smoke_test:
        smoke_test(work_dir, schedule)

    if args.no_eval:
        return 0

    if not args.task:
        print("[warn] no --task provided and --no-eval not set; nothing to do")
        return 0

    out_dir = Path(args.output_dir) / args.schedule
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(work_dir / "schedule_meta.json", out_dir / "schedule_meta.json")

    cmd = [
        sys.executable, "-m", "src.scripts.eval.launch_eval",
        "--model", str(work_dir),
        "--model-type", "hf",
        "--task", *args.task,
        "--output-dir", str(out_dir),
        "--batch-size", str(args.batch_size),
        "--gpus", str(args.gpus),
        "--model-args", "trust_remote_code=true",
    ]
    if args.num_shots is not None:
        cmd += ["--num-shots", str(args.num_shots)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    print(f"[info] running: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=os.environ.get("EMO_REPO_ROOT") or os.getcwd())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
