# Cardio-JEPA — fixes for the full run (forward prediction)

## What went wrong in `runs/full`

| # | Finding (from your `*_diagnostics.json` and screenshots) | Effect |
|---|---|---|
| 1 | `masked_only`, `forward_only` and `dual` each have **one epoch** of history and `"collapsed": true` at steps 1600 / 1600 / 1800. The collapse guard (`on_collapse="stop"`) killed all three. | Every Stage 1, Stage 2 and Stage 2c number came from models trained for about 0.6 % of the plan (1.8k of 301.7k steps). |
| 2 | The guard's reference loss was the value at **step 5** (≈0.99, untrained network). The loss drops 50–100× in the first ~1k steps on *every* arm, because at init the encoder output is offset-dominated (std ≈ 1e-3) and the LayerNormed targets are almost positional constants. Your `rq3_vq` run was flagged at step 1000 and **recovered** (std 0.0012 → 0.13 by epoch 5) because it ran with `warn`. | False positives that ended healthy runs early. |
| 3 | `full_config()` never set `anti_collapse="vicreg"` (only `kaggle_config()` did), so every Stage 1 arm in full mode was EMA-only. | No variance floor on the pooled representation. Stage 3's `ema` row is the stopped Stage 1 `dual` (std 0.0013, cos 1.0000, AUROC 0.640) compared against arms trained for 100 epochs. |
| 4 | The cross-window forecast target was the **per-patch** (C×P = 400-token) future latent. Which beat lands in which 0.2 s patch 10 s – 1 h ahead is unpredictable. | The best the predictor can do is output the average patch and ignore the context, which is why `cos_pred` ≈ 0.000 at every horizon. |
| 5 | Pooled latents are offset-dominated, so a raw regression loss is spent almost entirely on the constant offset. | Little gradient reaches the between-sample variation, which is the only part of the future that carries information. |
| 6 | Training horizons were {10, 30, 60, 300} s; Stage 2c also scored 900 s and 1800 s, and Stage 2/4 probed up to 3600 s. | The 600 s and 1800 s horizon embeddings never received a gradient. |
| 7 | `ForwardPairDataset` fixed 200 pairs per record at construction: 41,400 pairs. | About 1.5 % of your 4,573 h of Holter data was ever used for `L_fwd`, and the same pairs repeated every epoch. |
| 8 | `forecast_skill` compared the prediction (LayerNormed-target space) against the **raw** token mean, and built persistence from the **online** encoder. | Three different spaces. The persistence cosines of 0.01–0.04 for neighbouring windows of the same patient come from this, not from the data. |
| 9 | All long records were used in pretraining **and** as Stage 2/2c/4 test patients. | Optimistic anticipation and skill numbers. |
| 10 | VQ arm: cosine logits / temperature 0.5 means logits in [−2, 2]. The softmax over 512 codes is uniform (your log: perplexity 511.9). | Every target became the codebook mean, so `L_fwd`/`L_mask` fell to ~1e-4: a constant-target collapse that the variance metrics don't show. |
| 11 | Stage 4 trains on 12 context windows and back-propagates through all 12 per sample (768-d, 12 layers, batch 64). | CUDA out-of-memory on the 32 GB card. |
| 12 | `WARM_START = "auto"` picks up any `.pth` under `runs/`. | A rerun would silently warm-start from the collapsed checkpoints. |

## What the patch changes

**`config.py`**
* `full_config()`: `anti_collapse="vicreg"`, `reg_target="pooled"`, β = 0.1, `context_windows=6`, `grad_context_windows=2`.
* `ForwardSamplingConfig`: horizons `(10, 30, 60, 300, 600, 1800, 3600)` drawn log-uniformly, `cross_target="pooled"`, `standardize_target=True`, `residual_persistence=True`, `grad_context_windows`, `resample_pairs=True`.
* `OptimConfig`: `collapse_min_epochs=3`, `collapse_patience=5`, `amp_dtype="auto"` (bf16 on the 5090), `accum_steps`.
* `vq_temperature` 0.5 → 0.07.

**`models/cardio_jepa.py`, `models/forward_predictor.py`**
* Cross-window forecasting now predicts the **pooled** future latent, standardised per dimension with running statistics (`RunningStandardizer`). The same transform is used in training and evaluation.
* Residual head: `pred = a_h ⊙ persistence + delta`, where `a_h` is a per-horizon, per-dimension gate (init 1) and `delta` is zero-initialised. The model **starts exactly at persistence**, so it learns the change and how much to regress toward the mean at each horizon. A unit test checks this.
* The predictor gets a dedicated pooled query slot, so the intra-window per-patch path and the cross-window pooled path still share one predictor.
* `encode_context`: only the last `grad_context_windows` windows carry gradient (fixes the Stage 4 OOM and roughly halves wide-context cost).
* New logged values: `loss_fwd_cross`, `loss_fwd_intra`, and per-batch `fwd_skill` / `fwd_cos_pred` / `fwd_cos_persist`. The epoch line now prints `xwin=`, `intra=` and `skill=`, so you can see whether forecasting is working **while training**.
* New `target_latent()`; `forecast_latent(..., return_persistence=True)`.

**`engine.py`**
* The collapse guard waits `collapse_min_epochs`, measures its reference loss as a running mean over one epoch *after* the warm-up, and needs `collapse_patience` consecutive positive checks.
* bf16 autocast without GradScaler when supported; gradient accumulation; each logged key is averaged over the batches that produced it (cross-window losses were previously diluted by short-clip batches).

**`data/longform.py`, `data/build.py`**
* `ForwardPairDataset(resample=True, loguniform=True)`: a fresh record (weighted by duration), start time and horizon on every access.
* `holdout_long_records()` (record-level, stratified by AF burden) and `restrict_to_records()`.

**`eval/benchmarks.py`**
* `forecast_skill` scores everything in the model's target space, adds `mse_skill = 1 − MSE_pred / MSE_persist`, takes `record_ids` (use the held-out records), and batches inference.

Tests: 35 existing + 6 new, all pass (`python -m pytest tests -q`).
Mechanism check (synthetic Holter data, tiny CPU model, 360 steps): on **held-out** records `mse_skill` was +0.16 / +0.14 / +0.13 / +0.10 at 10 / 60 / 300 / 600 s. That only shows the head can learn the change from the present; it is not an ECG result.

## Notebook edits (`cardio_jepa_local.ipynb`)

### Config cell
```python
RUN_MODE = "full"
CFG = {"smoke": smoke_config, "kaggle": kaggle_config, "full": full_config}[RUN_MODE]()
CFG.data.input_root = INPUT_ROOT
CFG.out_dir = os.path.join(WORK, "runs")
CFG.optim.num_workers = 4 if RUN_MODE != "smoke" else 0
# DELETE the old line:  CFG.forward.context_windows = 2   (full_config sets 6)
CFG.optim.amp_dtype = "bf16"                 # RTX 5090
CFG.optim.on_collapse = "warn"               # never silently abandon a Stage-1 arm
CFG.hierarchical.enabled = False
```

### Warm-start cell
```python
WARM_START = "off"     # the old runs/full checkpoints are collapsed or stale; start fresh
```
Also move or rename the old `runs\full` folder before rerunning.

### Loader cell: hold out long records *before* pretraining
```python
from cardio_jepa.data import holdout_long_records, restrict_to_records

LONG_RECORDS = load_long_corpus(FOUND, CFG, verbose=True)
TRAIN_IDS, HELD_IDS = holdout_long_records(LONG_RECORDS, test_frac=0.3, seed=CFG.optim.seed)
print(f"long records: {len(TRAIN_IDS)} pretrain / {len(HELD_IDS)} held out")

LOADER, DATA_INFO = build_pretrain_loaders(
    FOUND, CFG, long_records=[LONG_RECORDS[i] for i in TRAIN_IDS],
    samples_per_record=200, verbose=True)
```
(`samples_per_record` now only sets the nominal epoch size; pairs are re-drawn every access.)

### Stage 2: split by the same held-out records
Replace the `split_by_record(...)` line with:
```python
tr = restrict_to_records(samples, TRAIN_IDS)
te = restrict_to_records(samples, HELD_IDS)
```
Do the same inside `horizon_sweep` (Stage 4) and for `btr, bte` in Stage 4b.

### Stage 2c: forecast skill on held-out records
```python
df = forecast_skill(entry["model"], LONG_RECORDS, CFG.signal, DEVICE,
                    horizons_s=(10.0, 60.0, 300.0, 900.0, 1800.0, 3600.0),
                    n_samples=1024, context_windows=CFG.forward.context_windows,
                    record_ids=HELD_IDS, seed=CFG.optim.seed, verbose=True)
```
Read **`mse_skill`** first (fraction of persistence error removed); `skill` (cosine) is secondary.
Positive values on held-out patients are the evidence for the world-model claim.

### Stage 3: fix the arm reuse
Stage 1 `dual` is now the **vicreg** arm, not `ema`. Change the reuse test to:
```python
reuse = (arm == CFG.loss.anti_collapse and (STAGE1.get("dual") or {}).get("model") is not None)
```
so `ema` is trained for the full schedule (with `on_collapse="warn"`) like the other arms.

### Stage 4: memory
At the very top of the notebook (before `import torch`):
```python
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```
In `build_stage4_cfg`:
```python
c.forward.context_windows = L2W
c.forward.grad_context_windows = 2
c.optim.batch_size = 16
c.optim.accum_steps = 4          # effective batch 64, same as Stage 1
```

## Compute budget on the 5090
Your `rq3_vicreg` checkpoints are ~815 s apart per epoch (3,017 steps), so about **22.6 h per 100-epoch model**.
With 9 models that is over a week. Suggested order:

1. **Stage 1 `dual` alone, 30 epochs** (~7 h). Watch `xwin` fall and `skill` turn positive in the epoch line; then run Stage 2c on the held-out records.
2. If skill is positive: `masked_only` and `forward_only` at the same 30 epochs (the controlled comparison), then Stage 2.
3. Final paper model: `dual`, 100 epochs.
4. Stage 3 / Stage 4 only after the above. Every arm must use the same epoch count, or the comparison is not fair.

## Your local package differs from the repo
Your traceback shows `pretrain(..., **CKPT_KW)` with resume / periodic-checkpoint arguments that are not in GitHub's `engine.py`.
`cardio_jepa.patch` applies cleanly to the GitHub version.
For your local `engine.py`, port these hunks by hand: `resolve_amp`, the `collapse_min_epochs` / `collapse_patience` guard, the `loss_ref` running mean, `accum_steps`, per-key `cnt` averaging, and the `"fwd_"` prefix in `monitor.log(extra=...)`.
The other files have no local changes visible in your traceback and can be replaced directly.
