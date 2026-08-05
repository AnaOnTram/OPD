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

| Teacher \ Student | YOLO11n | YOLO11m | YOLO26n | YOLO26m |
|----------------------|---------|---------|---------|---------|
| **Small (ViTs)** | ✓ | ✓ | ✓ | — |
| **Large (ViTl)** | ✓ | ✓ | ✓ | ✓ |
| **Huge (ViTh)** | ✓ | ✓ | ✓ | — |

### Student Model Profiles

| Model | Params | GFLOPs | Distilled `.pt` size | Detect loss family |
|-------|--------|--------|----------------------|--------------------|
| YOLO11n | 2.62 M | 6.6 | ~5.3–5.5 MB | dfl |
| YOLO26n | 2.57 M | 6.1 | ~5.2–5.5 MB | l1 |
| YOLO11m | ~20.1 M | ~68 | ~39–41 MB | dfl |
| YOLO26m | ~20 M class | — | ~44 MB | l1 |

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

**References (not distilled):**

| Run | Model | Best mAP@50 | Best mAP@50-95 | Final mAP@50 | Final mAP@50-95 |
|---|---|---:|---:|---:|---:|
| `yolo11n_stock` | YOLO11n from `yolo11n.pt` | 0.9787 | **0.7807** | 0.9791 | 0.7790 |
| `yolo11m_stock` | YOLO11m from `yolo11m.pt` | **0.9872** | **0.7920** | 0.9811 | 0.7914 |

> **Capacity-matched control:** distilled 11n must be compared to stock 11n; distilled 11m to stock 11m. Comparing distilled 11m only to stock 11n confounds **model size** with **distillation**.

### Key Observations

#### Distill vs stock (capacity-matched) — revised

| Comparison | Δ best mAP@50 | Δ best mAP@50-95 | Verdict |
|---|---:|---:|---|
| Best distilled 11m vs **stock 11m** | −0.03 pp (Large) … −0.64 pp (Huge) | −0.07 … +0.11 pp | **No meaningful distill lift** |
| Best distilled 11n vs **stock 11n** | −0.44 pp (Huge) | **−1.83 pp** | **Distill hurts nano** |
| Distilled 11m vs stock 11n | +0.8 pp | +1.1 pp | **Capacity only** (n→m), not distill |

#### Student capacity (Small teacher fixed)

| Student | Best mAP@50 | Best mAP@50-95 |
|---------|-------------|----------------|
| YOLO26n | 0.9651 | 0.7284 |
| YOLO11n | 0.9734 | 0.7575 |
| **YOLO11m** | **0.9846** | **0.7923** |

- Medium YOLO11m gains **~1.1 pp mAP@50** and **~3.5 pp mAP@50-95** over YOLO11n under the same Small teacher — **capacity**, not distill (stock 11m alone is 0.9872 / 0.7920).
- Small + YOLO11m does **not** beat **stock YOLO11m**; it only beats stock YOLO11n because of size.
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

- **Huge is the best distilled nano** (~+0.5 pp mAP@50-95 vs Small).
- All three remain **below stock YOLO11n** on mAP@50-95 (0.7807). Distill **hurts** nano; on medium it is a **tie** with stock 11m (not a win).

#### Teacher size (YOLO26n fixed, T mostly 0.07)

| Teacher | T | Best mAP@50 | Best mAP@50-95 | Distill local_loss (last 10) |
|---------|---|-------------|----------------|------------------------------|
| **Small** | 0.07 | **0.9651** | 0.7284 | **0.145** |
| Large | 0.10 | 0.9640 | 0.7336 | 0.414 |
| Huge | 0.07 | 0.9648 | **0.7355** | 1.088 |

- **mAP@50 is a three-way tie**; mAP@50-95 slightly favors larger teachers (~0.7 pp), still well below YOLO11n.

#### Medium family summary (incl. stock control)

| Rank (mAP@50-95) | Run | Best mAP@50 | Best mAP@50-95 |
|------------------|-----|-------------|----------------|
| — | **stock YOLO11m (no distill)** | **0.9872** | **0.7920** |
| 1 | Huge + YOLO11m | 0.9808 | 0.7931 |
| 2 | Small + YOLO11m | 0.9846 | 0.7923 |
| 3 | Large + YOLO11m | 0.9869 | 0.7913 |
| 4 | Large + YOLO26m | 0.9836 | 0.7815 |

Distilled mediums sit in a **±0.1 pp** band around stock 11m on mAP@50-95; stock owns best mAP@50.

#### Nano family summary

| Rank | Run | Best mAP@50-95 |
|------|-----|----------------|
| — | **stock YOLO11n (no distill)** | **0.7807** |
| 1 | Huge + YOLO11n | 0.7624 |
| 2 | Small + YOLO11n | 0.7575 |
| 3 | Large + YOLO11n | 0.7564 |
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

### 4.1 Distillation does not beat capacity-matched stock FT
With stock **YOLO11m** in the suite, the previous story that “distill helps medium” collapses: distilled 11m ≈ stock 11m (noise), distilled 11n < stock 11n. Any 11m-vs-11n gap is **capacity**.

### 4.2 Teacher-Student Capacity Match
On **YOLO11m**, all three teachers land within ~0.2 pp mAP@50-95 of each other **and** of stock 11m. On **YOLO11n**, Huge is the best distilled nano but still loses to stock. Teacher size is secondary; distill itself is not justified here.

### 4.3 Student Capacity Matters More Than Teacher Size (and more than distill)
Moving **n → m** under YOLO11 is the big YOLO win (~+1.1 pp mAP@50-95 stock-to-stock). Teacher sweeps only move sub-pp within a student size.

### 4.4 Architectural Alignment (11 vs 26)
Under the **same Large teacher**, YOLO11m beats YOLO26m by **~1.0 pp** mAP@50-95 with almost identical distill losses. Combined with YOLO11n ≫ YOLO26n, this suite favors the **YOLO11 family** after labeled fine-tune (with or without distill).

### 4.5 Temperature
Prefer **T=0.07** for new runs so teacher size is not confounded (Large+YOLO26n used 0.10). Large+YOLO11m and Large+YOLO26m both used 0.07 — fair medium architecture comparison.

### 4.6 Limited Distill Data
With ~1,050 unlabeled images, large teachers do not pull far ahead on final mAP; labeled fine-tune saturates YOLO heads.

---

## 5. Recommendation

| Goal | Suggestion |
|------|------------|
| **Best YOLO accuracy (default)** | **Stock YOLO11m** fine-tune (`yolo11m_stock`) — mAP@50 **0.9872**, mAP@50-95 **0.7920**; **skip distill** |
| **Best small edge YOLO** | **Stock YOLO11n** fine-tune — distilled nano is worse |
| **If already distilled (mAP@50)** | Large + YOLO11m ≈ stock 11m (0.9869) — keep only if already trained |
| **If already distilled (mAP@50-95)** | Any of Small/Large/Huge → 11m — all ≈ stock 11m |
| **YOLO26m vs YOLO11m** | Prefer **YOLO11m** (stock or distill) over YOLO26m on this data |
| **Avoid as default** | Distillation pipeline for this dataset; YOLO26n over YOLO11n |