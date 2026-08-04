# Distillation Experiment Analysis

**Date:** 2026-07-10 (updated 2026-08-04)

**Objective:** Compare DINOv3-to-YOLO distillation configurations and determine the best teacher-student combination for downstream object detection.

---

## 1. Experiment Configurations

| Run ID | Output Dir | Teacher | Student | Temperature | Distill | Detection fine-tune |
|--------|-----------|---------|---------|-------------|---------|---------------------|
| 1 | `out/distill_ViTs_yolo11n` | `dinov3/vits16plus` (Small) | YOLO11n | 0.07 | 100 ep | `runs/detect/train-ViTs_yolo11n` |
| 2 | `out/distill_ViTh_yolo26n` | `dinov3/vith16plus` (Huge) | YOLO26n | 0.07 | 100 ep | `runs/detect/train-ViTh_yolo26n` |
| 3 | `out/distill_ViTl_yolo26n` | `dinov3/vitl16` (Large) | YOLO26n | 0.10 | 100 ep | `runs/detect/train-ViTl_yolo26n` |
| 4 | `out/distill_ViTh_yolo11m` | `dinov3/vith16plus` (Huge) | YOLO11m | 0.07 | 100 ep | `runs/detect/train-ViTh_yolo11m` |
| 5 | `out/distill_ViTs_yolo11m` | `dinov3/vits16plus` (Small) | YOLO11m | 0.07 | 100 ep | `runs/detect/train-ViTs_yolo11m` |
| 6 | `out/distill_ViTl_yolo11n` | `dinov3/vitl16` (Large) | YOLO11n | 0.07 | 100 ep | `runs/detect/train-ViTl_yolo11n` |
| 7 | `out/distill_ViTs_yolo26n` | `dinov3/vits16plus` (Small) | YOLO26n | 0.07 | 100 ep | `runs/detect/train-ViTs_yolo26n` |
| 8 | `out/distill_ViTl_yolo26m` | `dinov3/vitl16` (Large) | YOLO26m | 0.07 | 100 ep | `runs/detect/train-ViTl_yolo26m` |
| 9 | `out/distill_ViTh_yolo11n` | `dinov3/vith16plus` (Huge) | YOLO11n | 0.07 | 100 ep | `runs/detect/train-ViTh_yolo11n` |
| 10 | `out/distill_ViTl_yolo11m` | `dinov3/vitl16` (Large) | YOLO11m | 0.07 | 100 ep | `runs/detect/train-ViTl_yolo11m` |

### Coverage matrix

| Teacher ↓ \ Student → | YOLO11n | YOLO11m | YOLO26n | YOLO26m |
|----------------------|---------|---------|---------|---------|
| **Small (ViTs)** | ✓ + mAP | ✓ + mAP | ✓ + mAP | — |
| **Large (ViTl)** | ✓ + mAP | ✓ + mAP | ✓ + mAP | ✓ + mAP |
| **Huge (ViTh)** | ✓ + mAP | ✓ + mAP | ✓ + mAP | — |

### Student Model Profiles

| Model | Params | GFLOPs | Distilled `.pt` size | Detect loss family |
|-------|--------|--------|----------------------|--------------------|
| YOLO11n | 2.62 M | 6.6 | ~5.3–5.5 MB | dfl |
| YOLO26n | 2.57 M | 6.1 | ~5.2–5.5 MB | l1 (this suite’s newer 26n runs) |
| YOLO11m | ~20.1 M | ~68 | ~39–41 MB | dfl |
| YOLO26m | ~20 M class | — | ~44 MB | l1 |

> **Note:** YOLO26n is not actually smaller than YOLO11n. Differences among nano students are architectural. Medium students are ~8× params vs nano. YOLO11m and YOLO26m are **not interchangeable** (different head/loss family).

---

## 2. Downstream Detection Results

Fine-tuned 100 epochs on `yolo11_Training` / labeled train set (imgsz 640). Older YOLO11m runs used batch 12–13; newer medium/nano runs typically batch 16.

| Run | Teacher | Student | Best mAP@50 | Epoch | Best mAP@50-95 | Epoch | Final mAP@50 | Final mAP@50-95 |
|---|---|---|---|---|---|---|---|---|
| **`train-ViTl_yolo11m`** | DINOv3-Large | **YOLO11m** | **0.9869** | 87 | 0.7913 | 100 | **0.9810** | 0.7913 |
| **`train-ViTs_yolo11m`** | DINOv3-Small | **YOLO11m** | 0.9846 | 83 | 0.7923 | 99 | 0.9806 | 0.7915 |
| **`train-ViTh_yolo11m`** | DINOv3-Huge | **YOLO11m** | 0.9808 | 98 | **0.7931** | 99 | 0.9802 | **0.7920** |
| `train-ViTl_yolo26m` | DINOv3-Large | YOLO26m | 0.9836 | 82 | 0.7815 | 90 | 0.9787 | 0.7815 |
| **`train-ViTh_yolo11n`** | DINOv3-Huge | **YOLO11n** | **0.9743** | 88 | **0.7624** | 99 | **0.9730** | **0.7622** |
| `train-ViTs_yolo11n` | DINOv3-Small | YOLO11n | 0.9734 | 88 | 0.7575 | 100 | 0.9707 | 0.7575 |
| `train-ViTl_yolo11n` | DINOv3-Large | YOLO11n | 0.9721 | 89 | 0.7564 | 99 | 0.9715 | 0.7552 |
| `train-ViTh_yolo26n` | DINOv3-Huge | YOLO26n | 0.9648 | 87 | 0.7355 | 100 | 0.9638 | 0.7355 |
| `train-ViTl_yolo26n` | DINOv3-Large | YOLO26n | 0.9640 | 87 | 0.7336 | 100 | 0.9636 | 0.7336 |
| `train-ViTs_yolo26n` | DINOv3-Small | YOLO26n | 0.9651 | 89 | 0.7284 | 100 | 0.9620 | 0.7284 |

**Reference (not distilled):** `yolo11n_stock` (COCO `yolo11n.pt` fine-tune) — best mAP@50 **0.9834**, best mAP@50-95 **0.7807**.

### Key Observations

#### Student capacity (Small teacher fixed)

| Student | Best mAP@50 | Best mAP@50-95 |
|---------|-------------|----------------|
| YOLO26n | 0.9651 | 0.7284 |
| YOLO11n | 0.9734 | 0.7575 |
| **YOLO11m** | **0.9846** | **0.7923** |

- Medium YOLO11m gains **~1.1 pp mAP@50** and **~3.5 pp mAP@50-95** over YOLO11n under the same Small teacher.
- Small + YOLO11m also beats **stock YOLO11n** fine-tune on both mAP@50 (+0.12 pp best) and mAP@50-95 (**+1.2 pp**).
- Small + YOLO26n remains the **weakest Small-teacher student** on mAP@50-95 (0.7284).

#### Teacher size (YOLO11m fixed, T=0.07) — complete sweep

| Teacher | Best mAP@50 | Best mAP@50-95 | Final mAP@50 | Final mAP@50-95 | Distill local_loss (last 10) |
|---------|-------------|----------------|--------------|-----------------|------------------------------|
| **Large** | **0.9869** | 0.7913 | **0.9810** | 0.7913 | 0.659 |
| Small | 0.9846 | 0.7923 | 0.9806 | 0.7915 | **0.129** |
| Huge | 0.9808 | **0.7931** | 0.9802 | **0.7920** | 1.026 |

- **Large wins mAP@50** (best in the whole suite: 0.9869) and final mAP@50.
- **mAP@50-95 is a three-way near-tie** (Huge 0.7931 ≈ Small 0.7923 ≈ Large 0.7913; ~0.2 pp spread) — within noise for this dataset.
- Distill local_loss ranks Small ≪ Large < Huge and **does not rank detection mAP**.
- **Practical picks for YOLO11m:**
  - **mAP@50 / “best box confidence”:** Large + YOLO11m
  - **mAP@50-95 / cheapest distill:** Small + YOLO11m (essentially same localization mAP, far lower distill cost than Large/Huge)

#### Large teacher fixed: YOLO11m vs YOLO26m (architecture control)

Same teacher (`vitl16`), same T=0.07, same protocol — only student family differs.

| Student | Export size | Detect loss | Best mAP@50 | Best mAP@50-95 | Final mAP@50-95 | Distill last-10 (train/local/global) |
|---------|-------------|-------------|----------------|----------------|-----------------|--------------------------------------|
| **YOLO11m** | ~40.6 MB | dfl | **0.9869** | **0.7913** | **0.7913** | 1.122 / 0.659 / 0.463 |
| YOLO26m | ~44.2 MB | l1 | 0.9836 | 0.7815 | 0.7815 | 1.070 / 0.647 / 0.423 |

| Metric | Δ (11m − 26m) |
|--------|----------------|
| Best mAP@50 | **+0.33 pp** |
| Best mAP@50-95 | **+0.98 pp** |
| Final mAP@50-95 | **+0.98 pp** |
| Distill local_loss | +0.012 (essentially equal fit) |

- Distill **feature-match quality is almost identical** (local ~0.65–0.66), so the gap is **not** “11m matched the teacher better.”
- Downstream detection clearly favors **YOLO11m**: ~1 pp mAP@50-95 and a smaller edge on mAP@50.
- YOLO26m remains competitive on mAP@50 but is **not a drop-in medium upgrade** vs YOLO11m on this data.
- Same pattern as nano: **YOLO11 family > YOLO26 family** after distill + fine-tune here (11n > 26n; 11m > 26m under Large).

#### Teacher size (YOLO11n fixed, T=0.07) — complete sweep

| Teacher | Best mAP@50 | Best mAP@50-95 | Final mAP@50-95 | Distill local_loss (last 10) |
|---------|-------------|----------------|-----------------|------------------------------|
| **Huge** | **0.9743** | **0.7624** | **0.7622** | 1.080 |
| Small | 0.9734 | 0.7575 | 0.7575 | **0.141** |
| Large | 0.9721 | 0.7564 | 0.7552 | 0.706 |

- **Huge is the best nano** (~+0.5 pp mAP@50-95 vs Small).
- All three remain **below stock YOLO11n** on mAP@50-95 (0.7807). Distill helps medium more than nano here.

#### Teacher size (YOLO26n fixed, T mostly 0.07)

| Teacher | T | Best mAP@50 | Best mAP@50-95 | Distill local_loss (last 10) |
|---------|---|-------------|----------------|------------------------------|
| **Small** | 0.07 | **0.9651** | 0.7284 | **0.145** |
| Large | 0.10 | 0.9640 | 0.7336 | 0.414 |
| Huge | 0.07 | 0.9648 | **0.7355** | 1.088 |

- **mAP@50 is a three-way tie**; mAP@50-95 slightly favors larger teachers (~0.7 pp), still well below YOLO11n.

#### Medium family summary

| Rank (mAP@50-95) | Run | Best mAP@50 | Best mAP@50-95 |
|------------------|-----|-------------|----------------|
| 1 | Huge + YOLO11m | 0.9808 | **0.7931** |
| 2 | Small + YOLO11m | 0.9846 | 0.7923 |
| 3 | **Large + YOLO11m** | **0.9869** | 0.7913 |
| 4 | Large + YOLO26m | 0.9836 | 0.7815 |

#### Nano family summary

| Rank | Run | Best mAP@50-95 |
|------|-----|----------------|
| 1 | **Huge + YOLO11n** | **0.7624** |
| 2 | Small + YOLO11n | 0.7575 |
| 3 | Large + YOLO11n | 0.7564 |
| — | stock YOLO11n (no distill) | 0.7807 |
| — | best YOLO26n (Huge) | 0.7355 |

---

## 3. Distillation-Phase Losses

Average of the last 10 distillation epochs:

| Run | `train_loss` | `local_loss` | `global_loss` | Avg Batch Time |
|---|---|---|---|---|
| DINOv3-Small + YOLO11n (0.07) | 0.785 | 0.141 | 0.644 | 0.32 s |
| **DINOv3-Small + YOLO11m (0.07)** | **0.565** | **0.129** | **0.435** | **0.62 s** |
| DINOv3-Small + YOLO26n (0.07) | 0.744 | 0.145 | 0.599 | 0.21 s |
| DINOv3-Large + YOLO11n (0.07) | 1.311 | 0.706 | 0.604 | 2.10 s |
| DINOv3-Large + YOLO11m (0.07) | 1.122 | 0.659 | 0.463 | 1.07 s |
| DINOv3-Large + YOLO26n (0.10) | 0.632 | 0.414 | 0.218 | 1.25 s |
| DINOv3-Large + YOLO26m (0.07) | 1.070 | 0.647 | 0.423 | 1.24 s |
| DINOv3-Huge + YOLO11n (0.07) | 1.795 | 1.080 | 0.715 | 0.44 s |
| DINOv3-Huge + YOLO11m (0.07) | 1.534 | 1.026 | 0.508 | 1.97 s |
| DINOv3-Huge + YOLO26n (0.07) | 1.826 | 1.088 | 0.738 | 2.59 s |

### Key Observations

- Small + YOLO11m has the **lowest total distill loss among T=0.07 runs**.
- **Large + YOLO11m** and **Large + YOLO26m** have nearly the same local/global losses; detection still splits them by ~1 pp mAP@50-95 → architecture, not match quality.
- All **Huge** students share high local loss (~1.03–1.09); Huge can still win on nano mAP@50-95.
- **Takeaway:** Distill loss is a useful fit/cost signal, but **final ranking must use detection mAP**.
- Batch times mix hardware; prefer loss and mAP over absolute step time across machines.

---

## 4. Interpretation

### 4.1 Teacher-Student Capacity Match
On **YOLO11m**, all three teachers land within ~0.2 pp mAP@50-95; Large uniquely tops **mAP@50**. On **YOLO11n**, Huge is a small but real win (~0.5 pp mAP@50-95). Teacher size is secondary to student family/capacity.

### 4.2 Student Capacity Matters More Than Teacher Size
Moving **n → m** under YOLO11 remains the big win (~+3 pp mAP@50-95). Teacher sweeps only move sub-pp within a student size.

### 4.3 Architectural Alignment (11 vs 26)
Under the **same Large teacher**, YOLO11m beats YOLO26m by **~1.0 pp mAP@50-95** with almost identical distill losses. Combined with YOLO11n ≫ YOLO26n, this suite favors the **YOLO11 family** for this data/protocol after DINOv3 distill + labeled fine-tune.

### 4.4 Temperature
Prefer **T=0.07** for new runs so teacher size is not confounded (Large+YOLO26n used 0.10). Large+YOLO11m and Large+YOLO26m both used 0.07 — fair medium architecture comparison.

### 4.5 Limited Distill Data
With ~1,050 unlabeled images, large teachers do not pull far ahead on final mAP; labeled fine-tune dominates.

---

## 5. Recommendation

| Goal | Suggestion |
|------|------------|
| **Best accuracy (mAP@50)** | **Large + YOLO11m** (`train-ViTl_yolo11m`) — suite best 0.9869 |
| **Best accuracy (mAP@50-95) / cost tradeoff** | **Small + YOLO11m** — tied localization mAP, cheapest distill among medium winners |
| **Best nano** | **Huge + YOLO11n**; **Small + YOLO11n** if distill cost matters |
| **YOLO26m vs YOLO11m** | Prefer **YOLO11m** under the same Large teacher (~1 pp mAP@50-95) |
| **Avoid as default** | YOLO26n over YOLO11n on this dataset |

### Still missing

- Optional: stock YOLO11m / YOLO26m baselines for pure “distill vs no-distill” medium controls
- Optional: Small/Huge + YOLO26m for a full 26m teacher sweep (only Large+26m exists)

### Suggested Next Steps

1. **Ship / default medium (mAP@50):** `runs/detect/train-ViTl_yolo11m/weights/best.pt`.
2. **Ship / default medium (cheap / mAP@50-95):** `runs/detect/train-ViTs_yolo11m/weights/best.pt`.
3. **Ship / default nano (accuracy):** `runs/detect/train-ViTh_yolo11n/weights/best.pt`.
4. **Ship / default nano (cheap distill):** `runs/detect/train-ViTs_yolo11n/weights/best.pt`.
5. **Optional baseline:** Fine-tune stock `yolo11m.pt` / `yolo26m` on the same data to measure distill lift for medium.
6. Optional: Small/Huge + YOLO26m if a full 26m teacher matrix is needed.

---

## 6. File Locations

- Distillation outputs:
  - `out/distill_ViTs_yolo11n` — Small + YOLO11n
  - `out/distill_ViTs_yolo11m` — Small + YOLO11m
  - `out/distill_ViTs_yolo26n` — Small + YOLO26n
  - `out/distill_ViTh_yolo11n` — Huge + YOLO11n
  - `out/distill_ViTh_yolo11m` — Huge + YOLO11m
  - `out/distill_ViTh_yolo26n` — Huge + YOLO26n
  - `out/distill_ViTl_yolo11n` — Large + YOLO11n
  - `out/distill_ViTl_yolo11m` — Large + YOLO11m
  - `out/distill_ViTl_yolo26n` — Large + YOLO26n
  - `out/distill_ViTl_yolo26m` — Large + YOLO26m
- Downstream detection runs:
  - `runs/detect/train-ViTs_yolo11m`, `train-ViTh_yolo11m`, `train-ViTl_yolo11m`
  - `runs/detect/train-ViTl_yolo26m`
  - `runs/detect/train-ViTs_yolo11n`, `train-ViTl_yolo11n`, `train-ViTh_yolo11n`
  - `runs/detect/train-ViTs_yolo26n`, `train-ViTh_yolo26n`, `train-ViTl_yolo26n`
  - Reference: `runs/detect/yolo11n_stock`
- Distillation script: `distill.py`
- Mermaid process diagrams: `/home/ross/distillation_process_mermaid.md`
