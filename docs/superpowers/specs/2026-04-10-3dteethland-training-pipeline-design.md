# 3DTeethLand Training Pipeline — Design Spec

**Date:** 2026-04-10
**Status:** Draft
**Goal:** Train accurate 3D landmark detection and deep tooth segmentation models using the ToothInstanceNet codebase on Google Colab Pro+ (A100), driven via Colab MCP from Claude Code.

---

## 1. Context

### What We Have
- **ToothInstanceNet repo** — a multi-stage 3D dental analysis pipeline (Align → Instance Segment → Landmarks)
- **3 pre-trained checkpoints**: `align.ckpt` (217MB), `instseg_full.ckpt` (111MB), `landmarks_full.ckpt` (116MB)
- **3DTeethLand dataset** — ~8.9GB compressed / ~31.3GB uncompressed across 7 data parts + landmark annotations
- **5-fold cross-validation splits** — `fold_0.txt` through `fold_4.txt`
- **Colab Pro+** — A100 GPU, 80GB VRAM, background execution, ~166GB disk
- **Colab MCP** — Claude Code drives the Colab session directly

### Architecture Summary
- **Backbone:** Stratified Transformer with CRPE attention + KPConv point embedding
- **Encoder:** 3 hierarchical scales with channels [48, 96, 192, 256], depths [3, 9, 3]
- **Landmark heads:** 6 output heads (seg, mesial_distal, facial, outer, inner, cusps) — each predicts distance + 3D offset
- **Loss:** BCELoss (segmentation) + LandmarkLoss (SmoothL1 + Chamfer + Separation)
- **Training:** AdamW, lr=0.0006, cosine annealing, 5-epoch warmup, gradient clip=35
- **Instance seg:** Spatial embedding method with learned Gaussian bandwidth clustering
- **Metrics:** Landmark F1 (Hungarian matching), mAP at [0.03, 0.06, 0.12, 0.17]mm thresholds

### Key Technical Details
- Requires **CUDA 12.1** + custom CUDA kernel compilation (`setup.py`)
- **PyTorch 2.3.0**, PyTorch Lightning 2.3.3, Python 3.10
- Config file at `teethland/config/config.yaml` — paths hardcoded to original author's machine, must be updated
- Data format: `.obj` meshes + `.json` segmentation labels + `__kpt.json` landmark annotations
- Organized as `(lower|upper)/PATIENT_ID/PATIENT_ID_(lower|upper).*`

---

## 2. Strategy: Evaluate First, Then Targeted Fine-Tuning

### Rationale
1. Pre-trained checkpoints are from the challenge authors (published at MICCAI 2025) — likely already competitive
2. Alignment is geometric and data-agnostic — almost certainly doesn't need retraining
3. Instance segmentation errors cascade to landmarks — must verify this stage is solid
4. Landmark detection is the end goal and most likely to benefit from fine-tuning
5. Fine-tuning from checkpoints converges in ~50-100 epochs vs 500 from scratch

### Decision Tree
```
Evaluate align checkpoint → Good? → Keep frozen
                          → Bad?  → Fine-tune (unlikely)

Evaluate instseg checkpoint → Good? → Keep frozen
                            → Bad?  → Fine-tune instseg first

Evaluate landmarks checkpoint → Baseline metrics recorded
                              → Fine-tune regardless (primary goal)
```

---

## 3. Phases

### Phase 1: Colab Environment Setup (~30-45 min)

**Objective:** Working environment with all dependencies, CUDA kernels compiled, data extracted.

**Steps:**
1. Connect to A100 runtime via Colab MCP
2. Verify GPU: `nvidia-smi` (A100, 80GB VRAM, CUDA 12.x)
3. Clone repo or upload from local
4. Create conda environment (Python 3.10)
5. Install pip requirements from `requirements.txt`
6. Compile CUDA kernels: `pip install -v -e .`
7. Mount Google Drive
8. Extract dataset to Colab local disk:
   ```
   /content/data/
   ├── lower/          # All lower jaw scans merged from parts 1-7
   │   └── PATIENT_ID/
   │       ├── PATIENT_ID_lower.obj
   │       └── PATIENT_ID_lower.json
   ├── upper/          # All upper jaw scans merged from parts 1-7
   │   └── PATIENT_ID/
   │       ├── PATIENT_ID_upper.obj
   │       └── PATIENT_ID_upper.json
   └── landmarks_train/
       ├── lower/
       │   └── PATIENT_ID/
       │       └── PATIENT_ID_lower__kpt.json
       └── upper/
           └── PATIENT_ID/
               └── PATIENT_ID_upper__kpt.json
   ```
9. Copy checkpoints to `/content/checkpoints/`
10. Update `config.yaml`:
    - `work_dir` → `/content/logs`
    - `datamodule.root` → `/content/data`
    - `datamodule.landmarks_root` → `/content/data/landmarks_train`
    - All `checkpoint_path` entries → `/content/checkpoints/...`
11. Verify data loading: instantiate TeethLandDataModule, call `setup('fit')`, print dataset sizes

**Success criteria:** `train.py landmarks --devices 1` starts without error (kill after 1 batch).

**Data extraction strategy:**
- Extract all 7 parts sequentially, merging into unified `/content/data/` structure
- Since parts have overlapping `lower/` and `upper/` directories, unzip with overwrite (`-o` flag)
- Delete zips after extraction to reclaim disk space
- Expected disk usage: ~31GB data + ~0.5GB checkpoints + ~5GB environment = ~37GB total

### Phase 2: Baseline Evaluation (~30 min)

**Objective:** Quantify pre-trained checkpoint performance to establish baselines.

**Steps:**
1. **Evaluate landmark checkpoint** on validation fold:
   - Load `landmarks_full.ckpt` into LandmarkNet
   - Run `trainer.validate(model, datamodule=dm)`
   - Record: Dice, IoU, Landmark F1, mAP at all thresholds
   
2. **Evaluate instance segmentation checkpoint:**
   - Load `instseg_full.ckpt` into DentalNet
   - Run validation
   - Record: Dice, IoU, ToothF1, FDI F1
   
3. **Run full inference pipeline** on a few samples:
   - Use `infer.py landmarks` with pre-trained checkpoints
   - Visually inspect output JSON files
   - Check landmark predictions against ground truth

4. **Record all baselines** in a structured format:
   ```
   Stage       | Dice  | IoU   | F1    | mAP
   ------------|-------|-------|-------|------
   Landmarks   | ?     | ?     | ?     | ?
   InstSeg     | ?     | ?     | ?     | N/A
   ```

**Success criteria:** Baseline numbers for all metrics. Decision on which stages need fine-tuning.

**Decision point:** Based on results:
- If instseg Dice > 0.90 and ToothF1 > 0.85 → keep frozen, focus on landmarks
- If instseg underperforms → add instseg fine-tuning to Phase 4
- Regardless → proceed to landmark fine-tuning

### Phase 3: Training Pipeline Validation (~1 hr)

**Objective:** Verify training works end-to-end with a short test run.

**Steps:**
1. **Configure for short test run:**
   - Modify epochs to 10
   - Keep all other hyperparameters as-is (lr=0.0006, warmup=5)
   - Use fold 0 for train/val split
   - batch_size=16 (adjust if OOM on A100)

2. **Run landmark training for 10 epochs:**
   ```python
   python train.py landmarks --devices 1
   ```

3. **Monitor during training:**
   - Loss curve: should decrease monotonically after warmup
   - Learning rate: verify warmup + cosine schedule
   - GPU utilization: should be >80%
   - Memory usage: track VRAM consumption
   - Batch throughput: samples/sec

4. **After 10 epochs, analyze:**
   - Training loss trajectory
   - Validation loss trajectory
   - Metric improvement over baseline
   - Any signs of instability (NaN, spikes, divergence)

5. **TensorBoard inspection:**
   - Loss curves (train/val)
   - Learning rate schedule
   - Per-head losses (seg, mesial_distal, facial, outer, inner, cusps)
   - Metric evolution

**Success criteria:** 
- Loss decreases during training
- No NaN or divergence
- Validation metrics improve or remain stable
- GPU memory fits within 80GB A100

**Adjustments if needed:**
- OOM → reduce batch_size to 8 or 4
- Slow convergence → increase lr to 0.001
- Instability → reduce gradient_clip_norm from 35 to 10

### Phase 4: Targeted Fine-Tuning (~4-8 hrs)

**Objective:** Fine-tune landmark model (and instseg if needed) for meaningful improvement.

**Steps:**

#### 4a. Landmark Fine-Tuning (Primary)
1. **Configure training:**
   - Start from `landmarks_full.ckpt` (resume training)
   - Epochs: 100 (with early stopping patience of 20)
   - lr: 0.0003 (halved from original — fine-tuning rate)
   - warmup_epochs: 3
   - All other config unchanged
   - Use fold 0 (or best fold from evaluation)
   - Enable checkpointing: save top-3 by val loss + top-3 by landmark F1

2. **Launch training:**
   ```python
   python train.py landmarks --devices 1 --checkpoint /content/checkpoints/landmarks_full.ckpt
   ```

3. **Monitoring schedule (via MCP):**
   - Every 10 epochs: check loss curves, metrics, learning rate
   - Every 25 epochs: compare against baseline — are we improving?
   - If val loss plateaus for 15 epochs: consider stopping early
   - If val loss increases for 10 epochs: overfitting — stop and use best checkpoint

4. **Auto-save to Google Drive:**
   - Sync checkpoints to Drive every 25 epochs
   - Save TensorBoard logs to Drive continuously
   - Save best checkpoint separately as `landmarks_finetuned_best.ckpt`

#### 4b. Instance Segmentation Fine-Tuning (If Needed)
- Only if Phase 2 shows instseg underperformance
- Start from `instseg_full.ckpt`
- Epochs: 50 (fine-tuning, not full training)
- lr: 0.0003
- Same monitoring schedule as 4a
- Must complete before re-running landmark training

**Success criteria:**
- Landmark F1 improves over Phase 2 baseline
- mAP improves at primary threshold (0.06mm)
- Val loss converges to stable minimum
- Best checkpoint saved to Drive

### Phase 5: Full Pipeline Evaluation (~30 min)

**Objective:** End-to-end evaluation with fine-tuned models.

**Steps:**
1. **Update config** to use fine-tuned checkpoints
2. **Run full inference pipeline** on validation set:
   ```python
   python infer.py landmarks --devices 1
   ```
3. **Compare results:**
   ```
   Metric          | Pre-trained | Fine-tuned | Delta
   ----------------|-------------|------------|------
   Landmark F1     | ?           | ?          | ?
   mAP@0.06mm      | ?           | ?          | ?
   Dice            | ?           | ?          | ?
   IoU             | ?           | ?          | ?
   ```
4. **Qualitative inspection:**
   - Load output `__kpt.json` files
   - Compare predicted landmarks to ground truth
   - Check for systematic errors (e.g., consistently missing cusp landmarks)
5. **Decision point:**
   - Results satisfactory → proceed to full training (Phase 6)
   - Specific issues identified → adjust hyperparameters and repeat Phase 4
   - Fundamental issues → rethink approach

### Phase 6: Full Training Run (Optional, ~12-24 hrs)

**Objective:** Train to convergence with optimal settings discovered in Phases 3-4.

**Steps:**
1. **Configure for full training:**
   - Epochs: 300-500 (from checkpoint)
   - Use best hyperparameters from Phase 4
   - Enable all augmentations
   - Consider cross-fold validation (train on folds 0-3, validate on fold 4)

2. **Launch as background execution** (Colab Pro+ feature)

3. **Monitoring (periodic MCP check-ins):**
   - Check every ~50 epochs
   - Track: loss, metrics, learning rate, GPU health
   - Early stopping if plateau detected

4. **Final outputs:**
   - Best checkpoint by validation loss
   - Best checkpoint by landmark F1
   - Best checkpoint by mAP
   - Full TensorBoard logs on Drive
   - Final metric report

---

## 4. Config Modifications

The `config.yaml` needs these changes for Colab:

```yaml
# Paths
work_dir: '/content/logs'
datamodule:
  root: '/content/data'
  landmarks_root: '/content/data/landmarks_train'

# Checkpoints
model:
  align:
    checkpoint_path: '/content/checkpoints/align.ckpt'
  instseg:
    checkpoint_path: '/content/checkpoints/instseg_full.ckpt'
  landmarks:
    checkpoint_path: '/content/checkpoints/landmarks_full.ckpt'

# Keep original training params
model:
  landmarks:
    lr: 0.0006          # Halve to 0.0003 for fine-tuning
    weight_decay: 0.0001
    epochs: 500          # Reduce for initial runs
    warmup_epochs: 5
```

**Note:** The config YAML has duplicate `checkpoint_path` keys per stage (only last one is read by `yaml.safe_load`). We'll clean this up to have exactly one path per stage.

---

## 5. Bug Fix Required

**`train.py:125` — Metric checkpoint mode is inverted for landmarks stage.**

```python
# Current (buggy):
mode='min' if stage in ['binseg', 'landmarks'] else 'max'

# Fix:
mode='min' if stage == 'binseg' else 'max'
```

`dice/val` is `BinaryF1Score()` (0-1, higher=better). Using `mode='min'` saves the **worst** checkpoints. This must be fixed before any training run.

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| CUDA kernel compilation fails on Colab | Use Dockerfile as reference for exact build args. Fallback: try PyTorch's built-in point cloud ops |
| Dataset too large for Colab disk | Extract one part at a time, delete zip after extraction. Use `teeth3ds_sample.zip` for initial testing |
| A100 OOM with batch_size=16 | Start with batch_size=8, increase if headroom. Monitor with `nvidia-smi` |
| MCP connection drops during training | Training continues in background (PL handles this). Reconnect and check TensorBoard logs |
| Overfitting during fine-tuning | Monitor train/val loss gap. Use early stopping. Keep augmentation enabled |
| Config YAML duplicate keys | Clean up config to have single path per stage before training |
| Pre-trained checkpoints don't load | Check PyTorch Lightning version compatibility. May need `strict=False` for partial loading |
| Colab session timeout | Use background execution (Pro+). Auto-save checkpoints to Drive every 25 epochs |

---

## 6. Deliverables

1. **Baseline metrics report** — pre-trained checkpoint performance on validation set
2. **Fine-tuned landmark model** — `landmarks_finetuned_best.ckpt` saved to Google Drive
3. **Fine-tuned instseg model** (if needed) — `instseg_finetuned_best.ckpt`
4. **Training logs** — full TensorBoard logs on Drive
5. **Comparison report** — pre-trained vs fine-tuned metrics
6. **Updated config.yaml** — cleaned up, portable paths

---

## 7. MCP Execution Plan

All phases driven via Colab MCP from Claude Code:

```
[Claude Code] --MCP--> [Colab Browser Tab] --Runtime--> [A100 GPU]
     |                                                        |
     |--- Execute cells ---------------------------------->   |
     |<-- Receive stdout/stderr --------------------------    |
     |--- Read files, check metrics --------------------->   |
     |<-- Return results ---------------------------------    |
```

**Monitoring approach:**
- Execute training in a cell
- Periodically run monitoring cells: read TensorBoard events, print latest metrics
- GPU health checks: `nvidia-smi` periodically
- Checkpoint sync: copy best checkpoints to Drive mount

**Drive structure:**
```
Google Drive/
└── 3DTeethLand/
    ├── data/           # Zip files uploaded here
    ├── checkpoints/    # Pre-trained + fine-tuned models
    └── logs/           # TensorBoard logs
```
