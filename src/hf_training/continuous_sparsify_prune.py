"""
Continuous Sparsification (CS) expert pruning for HuggingFace MoE models.

Implements the expert-selection variant of Savarese, Silva & Maire,
"Winning the Lottery with Continuous Sparsification" (NeurIPS 2020), mapped onto
EMO routed experts.

Core idea (vs. greedy_prune_layerwise / easy_ep / random):
  Instead of scoring experts from a passive forward pass and cutting by
  frequency / importance, CS attaches a learnable logit ``s_e`` to every routed
  (standard) expert, gates that expert's contribution by a temperature-scaled
  sigmoid ``sigma(beta * s_e)``, freezes ALL model weights, and trains only the
  mask logits on a small task calibration set with an L1 penalty on the soft
  gates. The temperature ``beta`` is annealed upward so the soft gate hardens
  toward a boolean decision. The learned decision is then baked into a
  *structural* prune via the existing ``prune_moe_layer_inplace`` kernel — so the
  emitted artifact (pruned model + pruning_metadata.json) is byte-identical in
  form to what the other methods produce and drops straight into the existing
  finetune / eval pipeline with zero inference overhead.

Shared (trailing) experts are never masked — their gate is fixed at 1.

Selection modes (how learned masks become the final cut):
  per_layer  — top (keep_k - num_shared) standard experts each layer (uniform keep_k).
  global     — rank all standard experts across MoE layers; keep a global budget
               of (keep_k - num_shared) * num_moe_layers (per-layer counts vary).
  threshold  — keep standard experts with s_e >= 0 (emergent count, pure CS).

Usage:
    python -m src.hf_training.continuous_sparsify_prune \\
        --model /path/to/model \\
        --task arc_challenge --split train \\
        --prune-keep-k 32 --num-shared-experts 1 \\
        --num-calibration 256 --num-steps 200 \\
        --selection-mode per_layer \\
        --save-path /tmp/pruned_cs
"""

import argparse
import json
import logging
import math
import os
import types
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.hf_training.data_utils import load_finetuning_dataset
from src.hf_training.finetune import MaskedLossDataCollator
from src.hf_training.greedy_prune_layerwise import (
    _capture_layer_output,
    _is_moe_layer,
    prune_moe_layer_inplace,
)

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monkeypatched MoE forward with CS mask injection
# ---------------------------------------------------------------------------
def _cs_moe_forward(self, hidden_states: torch.Tensor):
    """
    Re-implementation of ``EmoSparseMoeBlock.forward`` (modeling_emo.py:318) that
    multiplies each *selected* expert's routing weight by ``sigma(beta * s_e)``
    before the expert loop. Standard experts occupy indices
    ``[0, num_experts - num_shared_experts)``; shared experts occupy the trailing
    ``num_shared_experts`` indices and are gated at a fixed 1 (never pruned).

    Only the ``num_shared_experts >= 0`` paths are supported (matching easy_ep);
    the ``always_active_experts`` masked path raises NotImplementedError because
    its always-active semantics conflict with masking. The return signature
    ``(final_hidden_states, router_logits)`` and the ``norm_topk_prob`` branch are
    preserved.
    """
    if self.always_active_experts is not None and len(self.always_active_experts) > 0:
        raise NotImplementedError(
            "Continuous sparsification does not support the always_active_experts "
            "masked routing path; only num_shared_experts >= 0 is supported."
        )

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    if self.num_shared_experts > 0:
        router_logits_standard = router_logits[:, : -self.num_shared_experts]
        router_logits_shared = router_logits[:, -self.num_shared_experts :]

        routing_weights_standard = F.softmax(router_logits_standard, dim=1, dtype=torch.float)
        routing_weights_shared = F.softmax(router_logits_shared, dim=1, dtype=torch.float)

        routing_weights_standard, selected_experts_standard = torch.topk(
            routing_weights_standard, self.top_k - self.num_shared_experts, dim=-1
        )
        routing_weights_shared, selected_experts_shared = torch.topk(
            routing_weights_shared, self.num_shared_experts, dim=-1
        )

        routing_weights = torch.cat([routing_weights_standard, routing_weights_shared], dim=1)
        selected_experts = torch.cat(
            [
                selected_experts_standard,
                selected_experts_shared + (self.num_experts - self.num_shared_experts),
            ],
            dim=1,
        )
    else:
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

    if self.norm_topk_prob:
        if self.num_shared_experts > 0:
            raise NotImplementedError(
                "norm_topk_prob is not implemented for the case where num_shared_experts > 0"
            )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

    # --- CS mask injection -------------------------------------------------
    # full_mask: (num_experts,) — sigma(beta * s_e) for standard experts, 1 for
    # shared. Indexed by selected_experts (long) and applied to routing_weights
    # while still in float (before the cast to hidden_states.dtype) so the mask
    # gradient is not degraded by the bf16 round-trip.
    mask_std = torch.sigmoid(self.cs_beta * self.cs_mask_logits)  # (num_standard,) float
    if self.num_shared_experts > 0:
        full_mask = torch.cat(
            [
                mask_std,
                torch.ones(self.num_shared_experts, device=mask_std.device, dtype=mask_std.dtype),
            ]
        )
    else:
        full_mask = mask_std
    routing_weights = routing_weights * full_mask[selected_experts]

    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.num_experts
    ).permute(2, 1, 0)

    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def _attach_cs_masks(model, num_layers: int, s_init: float, beta_start: float):
    """
    For every MoE layer: register a trainable ``cs_mask_logits`` Parameter of shape
    ``(num_standard,)`` initialised to ``s_init`` (so sigma(beta_start * s_init) ~ 1
    — every expert starts fully on), store a mutable ``cs_beta`` float, and swap in
    the CS forward. Returns (moe_blocks, original_forwards) for later restore.
    """
    moe_blocks = []
    original_forwards = []
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        if not _is_moe_layer(layer):
            continue
        moe = layer.mlp
        aae = getattr(moe, "always_active_experts", None)
        if aae is not None and len(aae) > 0:
            raise NotImplementedError(
                f"Layer {layer_idx}: always_active_experts is set (len={len(aae)}). "
                "Continuous sparsification only supports the num_shared_experts>=0 path."
            )
        num_standard = moe.num_experts - moe.num_shared_experts
        dev = moe.gate.weight.device
        moe.cs_mask_logits = torch.nn.Parameter(
            torch.full((num_standard,), float(s_init), dtype=torch.float, device=dev)
        )
        moe.cs_beta = float(beta_start)
        original_forwards.append((moe, moe.forward))
        moe.forward = types.MethodType(_cs_moe_forward, moe)
        moe_blocks.append(moe)
    logger.info(f"Attached CS masks to {len(moe_blocks)} MoE layers")
    return moe_blocks, original_forwards


def _restore_forwards(original_forwards):
    for moe, fwd in original_forwards:
        moe.forward = fwd


def _beta_at_step(step: int, total_steps: int, beta_start: float, beta_end: float) -> float:
    """Exponential anneal from beta_start to beta_end across total_steps."""
    if total_steps <= 1:
        return beta_end
    frac = step / (total_steps - 1)
    return beta_start * (beta_end / beta_start) ** frac


def _manual_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal-LM cross-entropy with shifted labels and ignore_index=-100."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1).to(shift_logits.device),
        ignore_index=-100,
    )


def continuous_sparsify_prune(
    model_name: str,
    task_name: str,
    split: str,
    prune_keep_k: int,
    num_shared_experts: int,
    save_path: str,
    selection_mode: str = "per_layer",
    lambda_l1: float = 1e-4,
    lr: float = 0.1,
    num_steps: int = 200,
    beta_start: float = 1.0,
    beta_end: float = 200.0,
    s_init: float = 3.0,
    batch_size: int = 4,
    num_calibration: Optional[int] = None,
    max_seq_length: int = 4096,
    grad_clip: float = 1.0,
    device: Optional[str] = None,
    num_shots_override: Optional[int] = None,
    prune_seed: int = 0,
    trust_remote_code: bool = True,
) -> None:
    assert selection_mode in ("per_layer", "global", "threshold"), selection_mode

    logger.info(f"Loading model: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device is None else device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = config.num_hidden_layers
    logger.info(f"Model has {num_layers} layers")

    # Compute CE manually; suppress the model's router aux-loss path.
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False

    # --- Calibration data (with masked labels) --------------------------------
    logger.info(
        f"Loading calibration data: {task_name} ({split})"
        + (f" [num_shots={num_shots_override}]" if num_shots_override is not None else "")
    )
    dataset = load_finetuning_dataset(
        task_name, split, tokenizer, max_length=max_seq_length, num_shots_override=num_shots_override
    )
    if num_calibration is not None and num_calibration < len(dataset):
        logger.info(
            f"Loaded {len(dataset)} examples, subsampling to {num_calibration} (seed={prune_seed})"
        )
        g = torch.Generator().manual_seed(prune_seed)
        perm = torch.randperm(len(dataset), generator=g).tolist()
        dataset = dataset.select(perm[:num_calibration])
    else:
        logger.info(f"Loaded {len(dataset)} examples, using all")

    collator = MaskedLossDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(prune_seed),
    )

    # --- Attach CS masks & monkeypatch forward --------------------------------
    moe_blocks, original_forwards = _attach_cs_masks(model, num_layers, s_init, beta_start)
    if not moe_blocks:
        raise RuntimeError("No MoE layers found; nothing to sparsify.")

    # --- Freeze everything except the mask logits -----------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    mask_params = [blk.cs_mask_logits for blk in moe_blocks]
    for m in mask_params:
        m.requires_grad_(True)
    opt = torch.optim.Adam(mask_params, lr=lr)

    total_standard = sum(m.numel() for m in mask_params)
    logger.info(
        f"Calibrating {total_standard} mask logits across {len(moe_blocks)} layers "
        f"for {num_steps} steps (lambda={lambda_l1}, lr={lr}, "
        f"beta {beta_start}->{beta_end}, s_init={s_init})"
    )

    # --- Calibration loop -----------------------------------------------------
    model.train()  # enable grad graph; weights stay frozen via requires_grad=False
    data_iter = iter(loader)
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        beta = _beta_at_step(step, num_steps, beta_start, beta_end)
        for blk in moe_blocks:
            blk.cs_beta = beta

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        ce = _manual_ce(out.logits, labels)

        # L1 penalty on the soft gates of EVERY standard expert (so even
        # never-selected experts are pushed off).
        penalty = sum(torch.sigmoid(blk.cs_beta * blk.cs_mask_logits).sum() for blk in moe_blocks)
        penalty = lambda_l1 * penalty

        loss = ce + penalty
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mask_params, grad_clip)
        opt.step()

        if step % max(1, num_steps // 20) == 0 or step == num_steps - 1:
            with torch.no_grad():
                all_logits = torch.cat([m.detach().flatten() for m in mask_params])
                mean_gate = torch.sigmoid(beta * all_logits).mean().item()
                frac_on = (all_logits >= 0).float().mean().item()
            logger.info(
                f"step {step:4d}/{num_steps} beta={beta:7.2f} ce={ce.item():.4f} "
                f"penalty={penalty.item():.4f} mean_gate={mean_gate:.3f} "
                f"frac(s>=0)={frac_on:.3f}"
            )

    model.eval()

    # --- Selection -> keep-set ------------------------------------------------
    # final score per standard expert = s_e (sigma is monotonic in s_e).
    moe_layer_indices = [i for i in range(num_layers) if _is_moe_layer(model.model.layers[i])]
    # map layer_idx -> its mask block
    blk_by_layer = {}
    bi = 0
    for i in range(num_layers):
        if _is_moe_layer(model.model.layers[i]):
            blk_by_layer[i] = moe_blocks[bi]
            bi += 1

    experts_kept_per_layer: List[Optional[List[int]]] = [None] * num_layers
    keep_k_per_layer: List[Optional[int]] = [None] * num_layers

    def _shared_indices(moe):
        num_standard = moe.num_experts - moe.num_shared_experts
        return list(range(num_standard, moe.num_experts))

    if selection_mode == "global":
        # Rank all standard experts across MoE layers by s_e; keep a global budget.
        budget = (prune_keep_k - num_shared_experts) * len(moe_layer_indices)
        all_scores = []  # (score, layer_idx, local_expert_idx)
        for i in moe_layer_indices:
            blk = blk_by_layer[i]
            s = blk.cs_mask_logits.detach().float().cpu()
            for e in range(s.numel()):
                all_scores.append((s[e].item(), i, e))
        all_scores.sort(key=lambda t: t[0], reverse=True)
        kept_standard_by_layer = {i: [] for i in moe_layer_indices}
        for score, i, e in all_scores[:budget]:
            kept_standard_by_layer[i].append(e)
        for i in moe_layer_indices:
            blk = blk_by_layer[i]
            keep_standard = sorted(kept_standard_by_layer[i])
            experts_to_keep = keep_standard + _shared_indices(blk)
            experts_kept_per_layer[i] = experts_to_keep
            keep_k_per_layer[i] = len(experts_to_keep)
    else:
        for i in moe_layer_indices:
            blk = blk_by_layer[i]
            s = blk.cs_mask_logits.detach().float().cpu()
            num_standard = s.numel()
            if selection_mode == "per_layer":
                k_standard = min(prune_keep_k - num_shared_experts, num_standard)
                keep_standard = sorted(torch.topk(s, k_standard).indices.tolist())
            else:  # threshold
                keep_standard = sorted((s >= 0).nonzero(as_tuple=True)[0].tolist())
                if len(keep_standard) == 0:
                    # Never drop a whole layer: keep the single best expert.
                    keep_standard = [int(torch.argmax(s).item())]
            experts_to_keep = keep_standard + _shared_indices(blk)
            experts_kept_per_layer[i] = experts_to_keep
            keep_k_per_layer[i] = len(experts_to_keep)

    for i in moe_layer_indices:
        logger.info(
            f"Layer {i}: keep_k={keep_k_per_layer[i]} "
            f"({len(experts_kept_per_layer[i]) - blk_by_layer[i].num_shared_experts} standard "
            f"+ {blk_by_layer[i].num_shared_experts} shared)"
        )

    # --- Restore original forward before baking the structural prune ----------
    _restore_forwards(original_forwards)
    for blk in moe_blocks:
        # Drop the training-only scaffold so it is not saved into the checkpoint.
        if hasattr(blk, "cs_mask_logits"):
            del blk.cs_mask_logits

    # Build a representative batch for the before/after sanity check.
    sanity_batch = collator([dataset[j] for j in range(min(batch_size, len(dataset)))])
    sanity_inputs = {
        "input_ids": sanity_batch["input_ids"],
        "attention_mask": sanity_batch["attention_mask"],
    }

    # --- Bake structural prune ------------------------------------------------
    first_moe = moe_layer_indices[0]
    for i in moe_layer_indices:
        layer = model.model.layers[i]
        experts_to_keep = experts_kept_per_layer[i]
        layer_keep_k = keep_k_per_layer[i]
        target_shared = num_shared_experts if layer.mlp.num_shared_experts > 0 else 0

        if i == first_moe:
            hidden_before = _capture_layer_output(model, i, sanity_inputs)

        prune_moe_layer_inplace(layer, experts_to_keep, layer_keep_k, target_shared)

        if i == first_moe:
            hidden_after = _capture_layer_output(model, i, sanity_inputs)
            max_diff = (hidden_before - hidden_after).abs().max().item()
            if torch.allclose(hidden_before, hidden_after):
                raise RuntimeError(
                    f"Sanity check FAILED at layer {i}: hidden states identical before/after "
                    "pruning; the forward pass is not reflecting the pruned weights."
                )
            logger.info(
                f"Sanity check PASSED at layer {i}: "
                f"max hidden-state difference before/after pruning = {max_diff:.6f}"
            )

    # --- Update global config (mirror greedy_prune_layerwise_variable) --------
    actual_num_experts_per_layer = []
    actual_num_shared_experts_per_layer = []
    for i in range(num_layers):
        layer = model.model.layers[i]
        if _is_moe_layer(layer):
            actual_num_experts_per_layer.append(layer.mlp.num_experts)
            actual_num_shared_experts_per_layer.append(layer.mlp.num_shared_experts)
        else:
            actual_num_experts_per_layer.append(0)
            actual_num_shared_experts_per_layer.append(0)

    min_keep_k = min(k for k in keep_k_per_layer if k is not None)
    if hasattr(model.config, "num_experts_per_tok") and model.config.num_experts_per_tok > min_keep_k:
        model.config.num_experts_per_tok = min_keep_k
    model.config.num_experts = min_keep_k
    if hasattr(model.config, "num_local_experts"):
        model.config.num_local_experts = min_keep_k
    model.config.num_shared_experts = num_shared_experts
    model.config.output_router_logits = True
    model.config.num_experts_per_layer = actual_num_experts_per_layer
    model.config.num_shared_experts_per_layer = actual_num_shared_experts_per_layer

    # --- Save -----------------------------------------------------------------
    logger.info(f"Saving pruned model to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metadata = {
        "original_model": model_name,
        "prune_keep_k": prune_keep_k,
        "num_shared_experts": num_shared_experts,
        "pruning_method": "continuous_sparsification",
        "selection_mode": selection_mode,
        "task": task_name,
        "split": split,
        "num_calibration": num_calibration,
        "num_shots_override": num_shots_override,
        "prune_seed": prune_seed,
        "cs_hyperparams": {
            "lambda": lambda_l1,
            "lr": lr,
            "num_steps": num_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "s_init": s_init,
            "grad_clip": grad_clip,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
        },
        "keep_k_per_layer": keep_k_per_layer,
        "experts_kept_per_layer": experts_kept_per_layer,
    }
    with open(os.path.join(save_path, "pruning_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Done. Pruned model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Sparsification expert pruning for HuggingFace MoE models"
    )
    # Match existing prune scripts.
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prune-keep-k", type=int, required=True)
    parser.add_argument("--num-shared-experts", type=int, default=0)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-calibration", type=int, default=None)
    parser.add_argument("--num-shots", type=int, default=None)
    parser.add_argument("--prune-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    # CS-specific.
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="per_layer",
        choices=["per_layer", "global", "threshold"],
    )
    parser.add_argument("--lambda", dest="lambda_l1", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-end", type=float, default=200.0)
    parser.add_argument("--s-init", type=float, default=3.0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    continuous_sparsify_prune(
        model_name=args.model,
        task_name=args.task,
        split=args.split,
        prune_keep_k=args.prune_keep_k,
        num_shared_experts=args.num_shared_experts,
        save_path=args.save_path,
        selection_mode=args.selection_mode,
        lambda_l1=args.lambda_l1,
        lr=args.lr,
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        s_init=args.s_init,
        batch_size=args.batch_size,
        num_calibration=args.num_calibration,
        max_seq_length=args.max_seq_length,
        grad_clip=args.grad_clip,
        device=args.device,
        num_shots_override=args.num_shots,
        prune_seed=args.prune_seed,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
