"""Capture per-layer expert activation distributions on a calibration set.

Mirrors what ``greedy_prune_layerwise_variable`` measures (soft probability
mass per standard expert, averaged across calibration tokens) but runs
every layer in a single forward pass per batch since no layer is being
pruned mid-loop.

Outputs:
  * ``avg_probs.npy``         — float32 array (L, num_experts) with the
    per-token-averaged softmax-normalised standard-expert probabilities
    (last column is the always-active shared expert; it is 1.0 by
    construction since softmax over the 1-element shared set is trivial).
  * ``sorted_descending.npy`` — float32 array (L, num_experts) where row l
    holds ``avg_probs[l]`` sorted descending.
  * ``rank_at_keep_k.json``   — for each (k in --ranks-of-interest) the
    per-layer probability mass at that rank (1-indexed) of the descending
    list, plus the cumulative mass of the top-k.
  * ``activations_grid.png``  — 16-panel grid, one subplot per layer,
    sorted descending. Rank thresholds (default 31 standard + 32 incl.
    shared) marked vertically.
  * ``rank_marginal_freq.png``— single plot of the marginal-rank mass
    across layers.

Example:
    python -m scripts.selective_hf.dump_expert_activations \\
        --model allenai/Emo_1b14b_1T \\
        --task arc_challenge \\
        --output-dir /workspace/eval_runs_pool/activation_analysis/arc_challenge \\
        --batch-size 32
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.hf_training.data_utils import get_formatted_prompts

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_activations(
    model_name: str,
    task_name: str,
    split: str,
    output_dir: Path,
    batch_size: int = 32,
    num_calibration: Optional[int] = None,
    num_shots: Optional[int] = None,
    seed: int = 0,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    L = config.num_hidden_layers
    num_experts = config.num_experts
    num_shared = config.num_shared_experts
    num_standard = num_experts - num_shared
    logger.info(
        f"Config: L={L} num_experts={num_experts} num_shared={num_shared} "
        f"top_k={config.num_experts_per_tok}"
    )

    logger.info(f"Loading dataset: {task_name} ({split})")
    prompts, _ = get_formatted_prompts(task_name, split, num_shots_override=num_shots)
    if num_calibration is not None:
        n_keep = min(num_calibration, len(prompts))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(prompts), generator=g).tolist()
        prompts = [prompts[i] for i in perm[:n_keep]]
    logger.info(f"Using {len(prompts)} prompts")

    all_batches = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        all_batches.append(
            tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        )

    # Per-layer accumulators. Standard probs softmax'd over the 127 routed
    # experts; shared probs softmax'd over the 1 shared expert (always 1.0).
    tot_probs_standard = torch.zeros(L, num_standard, dtype=torch.float64)
    tot_probs_shared = torch.zeros(L, num_shared, dtype=torch.float64)
    tot_tokens = 0

    captured: dict = {}  # layer_idx -> tensor (B*T, num_experts), one batch at a time

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            # EmoSparseMoeBlock.forward returns (hidden_states, router_logits)
            router_logits = output[1] if isinstance(output, tuple) else output
            captured[layer_idx] = router_logits.detach()
        return hook

    handles = [
        model.model.layers[i].mlp.register_forward_hook(make_hook(i))
        for i in range(L)
    ]

    try:
        for batch in tqdm(all_batches, desc="Forward passes"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            captured.clear()
            with torch.no_grad():
                model(**batch, output_router_logits=False)
            mask = batch["attention_mask"]  # (B, T)
            B, T = mask.shape
            tot_tokens += int(mask.sum().item())
            for l in range(L):
                logits = captured[l]  # (B*T, num_experts) bf16 on GPU
                logits = logits.view(B, T, num_experts).float()
                std = F.softmax(logits[..., :num_standard], dim=-1)
                shr = F.softmax(logits[..., num_standard:], dim=-1)
                m = mask.unsqueeze(-1).float()
                tot_probs_standard[l] += (std * m).sum(dim=(0, 1)).cpu().to(torch.float64)
                tot_probs_shared[l] += (shr * m).sum(dim=(0, 1)).cpu().to(torch.float64)
    finally:
        for h in handles:
            h.remove()

    avg_standard = (tot_probs_standard / max(tot_tokens, 1)).numpy().astype(np.float32)
    avg_shared = (tot_probs_shared / max(tot_tokens, 1)).numpy().astype(np.float32)
    # Concat for a single (L, num_experts) array; shared experts are at the end.
    avg_probs = np.concatenate([avg_standard, avg_shared], axis=1)
    sorted_desc = np.sort(avg_probs, axis=1)[:, ::-1].copy()

    np.save(output_dir / "avg_probs.npy", avg_probs)
    np.save(output_dir / "sorted_descending.npy", sorted_desc)

    meta = {
        "model": model_name,
        "task": task_name,
        "split": split,
        "num_prompts": len(prompts),
        "num_tokens": tot_tokens,
        "batch_size": batch_size,
        "L": L,
        "num_experts": num_experts,
        "num_shared": num_shared,
        "num_standard": num_standard,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"Saved arrays to {output_dir}")
    return {"avg_probs": avg_probs, "sorted_desc": sorted_desc, "meta": meta}


def plot_grid(sorted_desc: np.ndarray, output_path: Path, ranks_of_interest: List[int]) -> None:
    L, E = sorted_desc.shape
    cols = 4
    rows = (L + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.4), sharex=True)
    axes = axes.ravel()
    rank_colors = ["#d62728", "#9467bd", "#2ca02c", "#ff7f0e"]
    for l in range(L):
        ax = axes[l]
        ax.bar(np.arange(E), sorted_desc[l], width=1.0, color="#4c78a8", linewidth=0)
        for k, c in zip(ranks_of_interest, rank_colors):
            if 1 <= k <= E:
                ax.axvline(k - 0.5, color=c, lw=1, linestyle="--",
                           label=f"rank {k}: {sorted_desc[l, k - 1]:.3g}")
        ax.set_title(f"layer {l}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
        if l % cols == 0:
            ax.set_ylabel("avg p(expert)", fontsize=8)
        if l >= L - cols:
            ax.set_xlabel("rank", fontsize=8)
    for j in range(L, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        "Per-layer expert activation distributions (sorted descending)",
        fontsize=11, y=1.0
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_marginal(sorted_desc: np.ndarray, output_path: Path, ranks_of_interest: List[int]) -> None:
    L, E = sorted_desc.shape
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    layers = np.arange(L)
    for k in ranks_of_interest:
        if 1 <= k <= E:
            ax.plot(layers, sorted_desc[:, k - 1], marker="o",
                    label=f"rank {k}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("avg p(expert)")
    ax.set_title("Activation mass at the marginal rank, per layer")
    ax.grid(alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-calibration", type=int, default=None)
    p.add_argument("--num-shots", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ranks-of-interest", type=str, default="31,32",
                   help="Comma-separated 1-indexed ranks to mark (default 31,32: "
                        "31 standard kept + 1 shared = 32 with uniform keep_k=32)")
    p.add_argument("--no-collect", action="store_true",
                   help="Skip collection; just re-plot from existing sorted_descending.npy")
    args = p.parse_args(argv)

    output_dir = Path(args.output_dir)
    ranks = [int(x) for x in args.ranks_of_interest.split(",") if x.strip()]

    if not args.no_collect:
        collect_activations(
            model_name=args.model,
            task_name=args.task,
            split=args.split,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_calibration=args.num_calibration,
            num_shots=args.num_shots,
            seed=args.seed,
        )

    sorted_desc = np.load(output_dir / "sorted_descending.npy")

    plot_grid(sorted_desc, output_dir / "activations_grid.png", ranks)
    plot_marginal(sorted_desc, output_dir / "rank_marginal_freq.png", ranks)

    rank_table = {
        "ranks": ranks,
        "values_per_layer": {
            f"rank_{k}": sorted_desc[:, k - 1].tolist() for k in ranks if 1 <= k <= sorted_desc.shape[1]
        },
        "cumulative_top_k": {
            f"top_{k}_mass": sorted_desc[:, :k].sum(axis=1).tolist() for k in ranks if 1 <= k <= sorted_desc.shape[1]
        },
    }
    (output_dir / "rank_at_keep_k.json").write_text(json.dumps(rank_table, indent=2))

    print(f"[done] artifacts in {output_dir}")
    for k in ranks:
        if 1 <= k <= sorted_desc.shape[1]:
            vals = sorted_desc[:, k - 1]
            print(f"  rank {k}: min={vals.min():.4f} max={vals.max():.4f} "
                  f"mean={vals.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
