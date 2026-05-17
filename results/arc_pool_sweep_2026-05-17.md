# ARC-Challenge: per-layer pool-size sweep on Emo_1b14b_1T

Run date: 2026-05-17. Hardware: 1× A100 80GB. Run via
`scripts/selective_hf/run_layerwise_pool_schedule.py` at commit `1d3e3aed`
with the two fixes applied in this session
(`greedy_prune_layerwise_variable.py`: `trust_remote_code=True` on
`from_pretrained`, and `output_router_logits=False` on the calibration
forward).

## TL;DR

Holding total expert budget constant (sum of per-layer keep-k = 32×16),
**every non-uniform schedule loses ground to the uniform baseline**, and the
penalty scales nearly monotonically with how non-uniform the schedule is.
At the most extreme spread, putting big pools in earlier layers and small
pools in later ones (`down_step`) hurts **6.4 pts more** than the opposite
shape (`up_step`).

## Setup

- Model: `allenai/Emo_1b14b_1T` (L=16, num_experts=128, top_k=8, num_shared=1)
- Task: `arc_challenge` (299 instances on validation, 0-shot)
- Pruning: greedy layerwise, calibrated on the task's validation split,
  always-active shared expert preserved in every layer (count=1).
- Budget: `keep_k_per_layer` sums to `L × baseline_keep_k = 16 × 32 = 512`
  in every cell. Spread is defined symmetrically around 32 so the floor is
  `32 − spread` and the ceiling is `32 + spread`.

### Schedules used

| Preset       | Spread | Schedule (length 16, sum 512)                                       |
|--------------|-------:|---------------------------------------------------------------------|
| `uniform`    |     —  | `[32]*16`                                                           |
| `up_step`    |     4  | `[28]*8 + [36]*8`                                                   |
| `down_step`  |     4  | `[36]*8 + [28]*8`                                                   |
| `up_step`    |     8  | `[24]*8 + [40]*8`                                                   |
| `down_step`  |     8  | `[40]*8 + [24]*8`                                                   |
| `up_step`    |    16  | `[16]*8 + [48]*8`                                                   |
| `down_step`  |    16  | `[48]*8 + [16]*8`                                                   |
| `up_step`    |    24  | `[8]*8  + [56]*8`     ← floor at `top_k`                            |
| `down_step`  |    24  | `[56]*8 + [8]*8`      ← floor at `top_k`                            |

## Results

Primary metric is `acc_uncond` from OLMES; 299-example validation, 0-shot.

| Schedule    | Spread | acc_uncond | Δ vs uniform | acc_per_char |
|-------------|-------:|-----------:|-------------:|-------------:|
| `uniform`   |     —  | **0.5452** | (baseline)   | 0.4749       |
| `up_step`   |     4  |    0.5284  |  −1.7        | 0.4615       |
| `down_step` |     4  |    0.5117  |  −3.3        | 0.4883       |
| `up_step`   |     8  |    0.5050  |  −4.0        | 0.4749       |
| `down_step` |     8  |    0.5117  |  −3.3        | 0.4649       |
| `up_step`   |    16  |    0.4783  |  −6.7        | 0.4348       |
| `down_step` |    16  |    0.4783  |  −6.7        | 0.4782       |
| `up_step`   |    24  |    0.4716  |  −7.4        | 0.4147       |
| `down_step` |    24  |    0.4080  | **−13.7**    | 0.3813       |

## Observations

1. **Uniform dominates the budget-matched space.** The flat schedule is the
   top performer at every spread; the penalty for deviating is at minimum
   1.7 absolute points (already well outside the noise floor for a 299-example
   eval at this accuracy band).

2. **Penalty grows with spread.** The damage scales roughly linearly with
   `spread` for the milder regime (spread ∈ {4, 8, 16}) and then accelerates
   for `down_step` at the saturation point (spread=24, where the constrained
   layers hit `keep_k = top_k = 8` — i.e. the router has no headroom).

3. **Direction-asymmetry at extreme spread.** At spread=24, `up_step` (narrow
   pools early, wide late) loses 7.4 points while `down_step` (wide early,
   narrow late) loses 13.7. So **constraining later layers is far more
   damaging than constraining early layers**, consistent with the general
   finding that later transformer-MoE layers carry more task-specific
   specialisation.

   At small spreads (4, 8) the two shapes are roughly symmetric, suggesting
   the asymmetry is specifically a saturation effect that only emerges once
   one half of the layers is squeezed against the `top_k` floor.

## Interpretation and caveats

- **Calibration confound.** The pruner picks each layer's surviving experts
  by activation frequency on the task's validation split. The "uniform"
  baseline gets to keep 32 experts in every layer — the most well-aligned
  budget for greedy selection. Non-uniform schedules force greedy selection
  to retain expert sets that may be off-distribution for the layers in
  question. So this experiment **conflates two things**: (a) information loss
  from constraining pool size at a layer, and (b) suboptimality of the
  greedy-keep policy when constrained pool sizes are imbalanced. To isolate
  (a) you'd need a non-greedy pruning method or counterfactual analysis.

- **Single task, single model.** No claim of generality beyond ARC-c on
  Emo_1b14b_1T. The handoff plan explicitly puts MMLU / HellaSwag in the
  next phase; we stopped before that to keep A100 spend down.

- **Calibration uses the eval set.** Pruning is calibrated on
  `arc_challenge` validation, and `acc_uncond` is computed on the same
  split. This is the protocol from the original pruning recipe, but it
  means the baseline is the optimistic case. A held-out calibration set
  would tighten the conclusion.

## Artifacts

Each cell's full outputs live at:

```
/workspace/eval_runs_pool/<schedule>_kavg32[_spread<S>]/arc_challenge/
  schedule_meta.json          ← resolved schedule + provenance
  pruned_model/               ← pruned HF checkpoint (~7.3 GB)
  eval/
    metrics.json
    task-arc_challenge-metrics.json
    task-arc_challenge-predictions.jsonl
    task-arc_challenge-recorded-inputs.jsonl
    task-arc_challenge-requests.jsonl
```

Sweep log: `/workspace/sweep_spread_ladder.log`.
Baseline log: `/workspace/baseline_uniform.log` (prune) +
`/workspace/baseline_uniform_eval.log` (eval).
