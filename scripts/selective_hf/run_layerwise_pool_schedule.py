"""Run HF eval on an EMO checkpoint pruned to a non-uniform per-layer pool size.

Pipeline per (schedule, task):
  1. Resolve the per-layer keep-k schedule (preset or explicit), enforcing
     floor = top_k (so top_k is preserved across layers) and ceiling =
     num_experts (the model's full pool).
  2. Call ``src.hf_training.greedy_prune_layerwise_variable`` to prune the
     model using calibration data from the task's validation split. The
     always-active shared expert is preserved in every layer.
  3. Call ``src.scripts.eval.launch_eval`` on the resulting pruned dir.

Each schedule keeps the always-active count (``num_shared_experts``) at 1 in
every layer. Only the pool size (``num_experts`` per layer) varies.

Example:
    python -m scripts.selective_hf.run_layerwise_pool_schedule \\
        --model allenai/Emo_1b14b_1T \\
        --task arc_challenge \\
        --schedule up_linear \\
        --baseline-keep-k 32 \\
        --output-dir /workspace/eval_runs_pool
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# -------- schedule construction --------

def _monotonic_int_schedule(L: int, target_sum: int, low: int, high: int) -> List[int]:
    """Non-decreasing integer schedule of length L summing to ``target_sum``.

    Starts from a linear interpolation low->high (rounded), then nudges
    individual entries until the sum matches while preserving non-decreasing
    order and the [low, high] bounds.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if not (low <= high):
        raise ValueError(f"low ({low}) must be <= high ({high})")
    if target_sum < low * L or target_sum > high * L:
        raise ValueError(
            f"target_sum={target_sum} infeasible for L={L}, low={low}, high={high}"
        )

    sched = [round(low + (high - low) * i / max(L - 1, 1)) for i in range(L)]
    diff = target_sum - sum(sched)

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


def resolve_schedule(
    name: str, L: int, baseline_keep_k: int, top_k: int, num_experts: int
) -> List[int]:
    """Resolve a preset name (or comma-separated list) to a per-layer pool schedule.

    Enforces: length == L, floor >= top_k (so top_k stays valid in every
    layer), ceiling <= num_experts, and for preset modes the schedule sums to
    ``baseline_keep_k * L``.
    """
    target_sum = L * baseline_keep_k

    if "," in name or name.startswith("["):
        cleaned = name.strip().lstrip("[").rstrip("]")
        sched = [int(x.strip()) for x in cleaned.split(",") if x.strip()]
        if len(sched) != L:
            raise ValueError(f"explicit schedule length {len(sched)} != L={L}")
        if min(sched) < top_k:
            raise ValueError(
                f"explicit schedule has min={min(sched)} < top_k={top_k}; "
                "the pruner would silently reduce top_k"
            )
        if max(sched) > num_experts:
            raise ValueError(
                f"explicit schedule has max={max(sched)} > num_experts={num_experts}"
            )
        return sched

    if baseline_keep_k < top_k:
        raise ValueError(
            f"baseline_keep_k ({baseline_keep_k}) must be >= top_k ({top_k}) "
            "so the floor of any schedule stays >= top_k"
        )
    if baseline_keep_k > num_experts:
        raise ValueError(
            f"baseline_keep_k ({baseline_keep_k}) must be <= num_experts ({num_experts})"
        )

    low = top_k
    # Symmetric spread around baseline: distance below the mean = distance above.
    spread = baseline_keep_k - low
    high = min(baseline_keep_k + spread, num_experts)

    if name == "uniform":
        return [baseline_keep_k] * L

    if name == "up_step":
        if L % 2 != 0:
            raise ValueError("up_step requires even L")
        # [low]*L/2 + [hi]*L/2 with average = baseline_keep_k
        hi = 2 * baseline_keep_k - low
        if hi > num_experts:
            raise ValueError(
                f"up_step would need high={hi} > num_experts={num_experts}"
            )
        return [low] * (L // 2) + [hi] * (L // 2)

    if name == "down_step":
        return list(reversed(resolve_schedule("up_step", L, baseline_keep_k, top_k, num_experts)))

    if name == "up_linear":
        return _monotonic_int_schedule(L, target_sum, low=low, high=high)

    if name == "down_linear":
        return list(reversed(_monotonic_int_schedule(L, target_sum, low=low, high=high)))

    raise ValueError(f"unknown schedule preset: {name}")


# -------- config readers --------

def _read_config_baseline(model_path_or_id: str) -> Tuple[int, int, int]:
    """Return (L, num_experts, top_k) from the model's config.json.

    Works both for HF Hub repo ids and local snapshot dirs. Only the config
    file is touched — no weights are downloaded.
    """
    p = Path(model_path_or_id)
    if p.exists() and p.is_dir():
        cfg_path = p / "config.json"
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = Path(hf_hub_download(model_path_or_id, "config.json"))
    with open(cfg_path) as f:
        cfg = json.load(f)
    L = int(cfg["num_hidden_layers"])
    num_experts = int(cfg["num_experts"])
    top_k = int(cfg["num_experts_per_tok"])
    return L, num_experts, top_k


# -------- main --------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HF hub repo id or local snapshot path (e.g. allenai/Emo_1b14b_1T)")
    p.add_argument("--task", required=True,
                   help="Eval task name (also used as calibration source). One task per run; "
                        "calibration is task-specific.")
    p.add_argument("--schedule", required=True,
                   help="Preset name (uniform, up_linear, down_linear, up_step, down_step) "
                        "or comma-separated explicit list of length num_hidden_layers")
    p.add_argument("--baseline-keep-k", type=int, default=32,
                   help="Average per-layer pool size; sum target is baseline_keep_k * L (default 32)")
    p.add_argument("--output-dir", required=True,
                   help="Root output dir; per (schedule, task) subdirs are created beneath")
    p.add_argument("--num-shared-experts", type=int, default=1,
                   help="Always-active shared expert count, same in every layer (default 1)")
    p.add_argument("--prune-split", default="validation",
                   help="Dataset split used for calibration (default validation)")
    p.add_argument("--num-calibration", type=int, default=None,
                   help="Subsample calibration set to this many prompts (default all)")
    p.add_argument("--prune-batch-size", type=int, default=32,
                   help="Forward-pass batch size for calibration (default 32)")
    p.add_argument("--prune-seed", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--num-shots", type=int, default=None,
                   help="Forwarded to both the pruner and launch_eval")
    p.add_argument("--limit", type=float, default=None,
                   help="Subsample fraction or count, forwarded to launch_eval")
    p.add_argument("--no-prune", action="store_true",
                   help="Reuse an existing pruned dir at the expected path; skip step 2")
    p.add_argument("--no-eval", action="store_true",
                   help="Stop after pruning; skip step 3")
    p.add_argument("--print-schedule-only", action="store_true",
                   help="Resolve and print the schedule, then exit")
    args = p.parse_args(argv)

    L, num_experts, top_k = _read_config_baseline(args.model)
    print(f"[info] model baseline: L={L}, num_experts={num_experts}, top_k={top_k}")

    schedule = resolve_schedule(args.schedule, L, args.baseline_keep_k, top_k, num_experts)
    print(f"[info] schedule '{args.schedule}': {schedule}")
    print(f"[info]   sum={sum(schedule)}  target={L*args.baseline_keep_k}  "
          f"min={min(schedule)}  max={max(schedule)}")

    if args.print_schedule_only:
        return 0

    # Sanitize schedule name for filesystem use.
    safe_name = args.schedule.replace(",", "_").replace("[", "").replace("]", "")
    run_dir = Path(args.output_dir) / f"{safe_name}_kavg{args.baseline_keep_k}" / args.task
    pruned_dir = run_dir / "pruned_model"
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Provenance.
    meta = {
        "model": args.model,
        "task": args.task,
        "schedule_name": args.schedule,
        "schedule": schedule,
        "baseline_keep_k": args.baseline_keep_k,
        "target_sum": L * args.baseline_keep_k,
        "L": L,
        "num_experts_full": num_experts,
        "top_k": top_k,
        "num_shared_experts": args.num_shared_experts,
        "prune_split": args.prune_split,
        "num_calibration": args.num_calibration,
        "prune_seed": args.prune_seed,
    }
    with open(run_dir / "schedule_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    repo_root = os.environ.get("EMO_REPO_ROOT") or os.getcwd()

    if not args.no_prune:
        if pruned_dir.exists() and any(pruned_dir.iterdir()):
            print(f"[warn] {pruned_dir} already non-empty; pass --no-prune to reuse, or remove it")
            return 2
        pruned_dir.mkdir(parents=True, exist_ok=True)
        prune_cmd = [
            sys.executable, "-m", "src.hf_training.greedy_prune_layerwise_variable",
            "--model", args.model,
            "--task", args.task,
            "--split", args.prune_split,
            "--keep-k-per-layer", ",".join(str(k) for k in schedule),
            "--num-shared-experts", str(args.num_shared_experts),
            "--save-path", str(pruned_dir),
            "--batch-size", str(args.prune_batch_size),
            "--prune-seed", str(args.prune_seed),
        ]
        if args.num_calibration is not None:
            prune_cmd += ["--num-calibration", str(args.num_calibration)]
        if args.num_shots is not None:
            prune_cmd += ["--num-shots", str(args.num_shots)]
        print(f"[step 1] pruning: {' '.join(prune_cmd)}")
        rc = subprocess.call(prune_cmd, cwd=repo_root)
        if rc != 0:
            print(f"[error] pruning failed with rc={rc}")
            return rc

    if args.no_eval:
        return 0

    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        sys.executable, "-m", "src.scripts.eval.launch_eval",
        "--model", str(pruned_dir),
        "--model-type", "hf",
        "--task", args.task,
        "--output-dir", str(eval_dir),
        "--batch-size", str(args.eval_batch_size),
        "--gpus", str(args.gpus),
        "--model-args", "trust_remote_code=true",
    ]
    if args.num_shots is not None:
        eval_cmd += ["--num-shots", str(args.num_shots)]
    if args.limit is not None:
        eval_cmd += ["--limit", str(args.limit)]

    print(f"[step 2] eval: {' '.join(eval_cmd)}")
    rc = subprocess.call(eval_cmd, cwd=repo_root)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
