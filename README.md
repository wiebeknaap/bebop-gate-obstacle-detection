# Enhanced Gate Detector

A real-time, classical computer-vision gate detector for autonomous drone racing. Detects competition gates defined by two blue vertical posts flanking a square opening, with a checkered black/white top bar. Designed to run on embedded ARM hardware without deep learning.

---

## Overview

The detector is "blue-first, geometry-aware": it uses colour segmentation to dramatically reduce the search space, then validates candidate post pairs with 13 independent scoring terms. It has been validated on full competition sequences including outdoor transition scenes, people-in-frame scenarios, and indoor competition hall footage.

```
Frame → preprocess → colour mask → blob candidates → post-pair hypothesis → scoring → GateDetection
                                        ↑                      ↑
                                  Hough verticals      13-term weighted score
                                        ↑
                                 LK tracker fallback
```

---

## Validation results

Evaluated on ~66 frames from three distinct scene types: outdoor (no gate), outdoor-to-indoor transition, and indoor competition hall.

| Metric | Value |
|--------|-------|
| Precision | 0.81 |
| Recall | 0.94 |
| F1 score | 0.87 |
| False positive rate | 0.21 |
| Avg TP confidence | 0.70 ± 0.03 |
| Avg checker score (gate) | 0.12 ± 0.03 |
| Avg colour purity (gate posts) | 0.76 ± 0.07 |

**Confirmed true positives:** 30 frames with the gate correctly bounded.  
**Confirmed false positives:** 7 frames — all caused by a person wearing blue military-pattern clothing walking past the shelf area. This is the primary residual failure mode.  
**False negatives:** 2 frames — gate at extreme viewing angle (yaw > 40°) where both posts were partially occluded.

### Runtime (640 × 480, x86-64 laptop)

| Configuration | ms/frame | fps |
|---|---|---|
| Full pipeline (default) | 118 ms | 8 fps |
| Multiscale disabled | 70 ms | 14 fps |
| Multiscale + Hough disabled | 56 ms | 18 fps |

On Cortex-A57 (typical drone compute): estimated 3–6 fps full pipeline, 8–12 fps with multiscale disabled. The Kalman-free LK tracker adds ~0.3 ms.

---

## Installation

```bash
pip install opencv-python numpy
```

No other dependencies. The file `gate_detector.py` is self-contained.

---

## Quick start

```python
import cv2
from gate_detector import EnhancedGateDetector, GateDetectorConfig

detector = EnhancedGateDetector()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed, mask, detection = detector.detect(frame)

    if detection.is_valid:
        print(f"Gate found! confidence={detection.confidence:.2f}")
        print(f"  lateral error: {detection.lateral_error_norm:+.3f}")
        print(f"  vertical error: {detection.vertical_error_norm:+.3f}")
        print(f"  yaw proxy: {detection.yaw_proxy:+.3f}")

    annotated = detector.annotate(processed, detection, mask)
    cv2.imshow("Gate Detector", annotated)
    if cv2.waitKey(1) == ord('q'):
        break
```

### Running on a folder of images

```bash
python gate_detector.py /path/to/images/ \
    --save-dir /path/to/output/ \
    --min-confidence 0.50 \
    --gate-width-m 1.4 \
    --fx-px 560
```

---

## Architecture

### Classes

| Class | Role |
|---|---|
| `GateDetectorConfig` | All hyperparameters in one dataclass; fully documented |
| `GateDetection` | Output struct: bounding boxes, confidence, control errors, optional pose |
| `EnhancedGateDetector` | Main detector; owns all sub-components |
| `BlueColorModel` | Online HSV histogram back-projection; adapts to scene illuminant |
| `TemporalMaskFusion` | Exponential-decay mask accumulator; suppresses transient noise |
| `LKGateTracker` | Lucas-Kanade sparse optical flow on gate corners; fallback when detection fails |
| `AlphaBetaGateTracker` | Legacy constant-velocity tracker; kept for API compatibility |
| `TrackerState` | Tracker output dataclass |

---

### Pipeline in detail

#### 1. Preprocess
- Optional portrait-frame rotation
- CLAHE contrast enhancement (LAB colour space, clip 2.2, 8×8 tiles)
- 3×3 Gaussian blur

#### 2. Colour segmentation — `segment_gate_blue()`

Three independent tests are AND/OR-combined:

| Test | Condition | Why |
|---|---|---|
| HSV band | H ∈ [85, 145], S > 35, V > 18 | Blue hue range in HSV is illuminant-stable |
| RGB dominance | B − G > 18 AND B − R > 12 | Rejects grey regions that sit inside the HSV band |
| Blue ratio | B / (R+G+B) > 0.34 | Catches low-saturation blue at the cost of some noise |

The adaptive illuminant correction shifts the hue band by ±14 units using grey-world estimation, handling warm artificial lighting without a calibration target.

After bitwise combination, four morphological operations run in sequence:
1. Median blur (5×5) — salt-and-pepper noise
2. Opening (3×3) — isolated pixels
3. Vertical closing (11×3) — reconnects pole fragments split by shadow
4. Horizontal closing (3×11) — thickens thin poles at distance

The bottom 5% of the frame is zeroed to suppress floor-mat clutter.

#### 3. Mask density gate

If more than 28% of the frame is blue, hypothesis testing is skipped. This handles frames where the drone is flying directly through a blue banner or wall.

#### 4. Vertical blob candidates

`cv2.connectedComponentsWithStats` followed by aspect, size, and fill filters:

| Filter | Value | Rationale |
|---|---|---|
| `min_component_area` | 100 px² | Eliminates noise blobs |
| `min_vertical_aspect` | 1.5 | Gate posts are taller than wide; square blobs are furniture/objects |
| `max_post_aspect` | 14.0 | Thin rods, person's legs have aspect 15–30 |
| `min_vertical_height_ratio` | 5% of frame | Posts must be tall enough to be real |
| Min fill density | 16% | Sparse blobs are usually noise artefacts |

#### 5. Hough vertical supplement — `_hough_vertical_segments()`

Probabilistic Hough (`cv2.HoughLinesP`) on a vertically-dilated version of the mask finds line segments even in fragmented masks (pole in partial shadow). Segments within 14 px horizontally are clustered into single post candidates.

**Column-density filter (key FP suppressor):** Each Hough cluster must have at least 18% of its vertical span containing blue pixels in a 5-pixel-wide window at its center-x. This separates real fragmented posts (col_fraction 0.75–0.95) from Hough edges fabricated at the boundary of horizontal blobs like shelves (col_fraction 0.05–0.12).

#### 6. Post-pair hypothesis — `_detect_post_pair_hypothesis()`

All O(n²) pairs of vertical candidates are tested. Hard geometry cuts run first:

| Filter | Value | Kills |
|---|---|---|
| Opening gap | 12 px – 82% frame width | Pairs too close or too far |
| `min_gap_to_post_width` | 1.5× max post width | Single pole split by segmentation |
| `max_post_aspect` per-pair | 14.0 | Repeated check after merge |
| `max_opening_aspect` | 2.2 | Wide banners (real opening ≈ square) |
| `max_outer_aspect` | 2.0 | Landscape-oriented false structures |
| Opening size | 7% × 9% of frame | Sub-threshold at distance |

Then a 13-term weighted score is computed:

| Term | Weight | Description |
|---|---|---|
| `symmetry` | 0.14 | Height and width ratio of the two posts |
| `overlap` | 0.13 | Vertical overlap fraction of the pair |
| `gap` | 0.11 | Gap/height ratio centred on 0.70 |
| `aspect` | 0.09 | Perspective-corrected opening aspect (yaw-aware) |
| `top_support` | 0.10 | Checkered top bar via 90th-percentile column gradient |
| `hollow` | 0.06 | Opening is clear of blue (gate interior is empty) |
| `corner_response` | 0.05 | Shi-Tomasi eigenvalue at four predicted corners |
| `gradient_perp` | 0.04 | Sobel-X alignment at post inner edges |
| `profile` | 0.07 | Scanline profile: post | gap | post contrast |
| `color_purity` | 0.10 | Raw HSV blue fraction of post pixels (orange/cream veto) |
| `isolation` | 0.08 | Blue does not extend past posts (banner veto) |
| `post_fill` | 0.02 | Mean fill density of both posts |
| `balance` + `center` | 0.02 | Left/right symmetry and frame centering |

Soft penalties (not hard cuts) are subtracted for: low vertical overlap, asymmetric heights, partially hollow opening, low colour purity, low isolation.

A candidate is **valid** if the total score ≥ `min_gate_confidence` (default 0.50).

#### 7. Hard vetoes

Two hard vetoes run inside the scoring loop:

**Colour purity veto:** For each post, the fraction of chromatic pixels (S≥28, V≥20) with hue in [75, 155] is measured directly from raw HSV — independent of the mask or the adaptive model. If either post has fewer than 8% blue chromatic pixels, the pair is rejected regardless of geometry score. This eliminates orange and cream pillars.

**Isolation veto:** The blue mask is sampled in windows just outside each post's outer edge at three heights. If the isolation score falls below 0.30, the pair is rejected. This eliminates blue banners where the colour continues past the "posts".

#### 8. Multi-scale pass

An independent half-resolution detection pass runs in parallel. At distance, thin posts aggregate into wider blobs at half-scale. The higher-confidence result between the two scales is used. Can be disabled with `enable_multiscale=False` for ~40% speed gain.

#### 9. Temporal tracking — `LKGateTracker`

When detection succeeds, Lucas-Kanade sparse optical flow (`cv2.calcOpticalFlowPyrLK`) is seeded at the four detected corners. In subsequent frames where detection fails (fast manoeuvre, brief occlusion), the tracker propagates corner positions through a 3-level pyramid with 21×21 windows.

Key guards:
- **Seeding guard:** LK is only seeded after `lk_seed_min_consecutive=2` consecutive valid detections. Prevents a single-frame false positive from generating a multi-frame ghost track.
- **Miss reset:** After `tracker_reset_missed_frames=15` consecutive misses, the tracker is cleared and temporal masks are reset.
- Tracker output is marked `is_valid=False` and mode `lk_tracked` — it guides the aircraft but does not count as a valid detection for downstream logic.

---

## Output: `GateDetection`

| Field | Type | Description |
|---|---|---|
| `is_valid` | bool | Score ≥ `min_gate_confidence` |
| `confidence` | float | Weighted score [0, 1] |
| `mode` | str | `"post_pair"`, `"lk_tracked"`, `"none"` |
| `center_px` | (float, float) | Opening centre in pixels |
| `outer_bbox` | (x, y, w, h) | Bounding box including posts |
| `opening_bbox` | (x, y, w, h) | Inner opening between posts |
| `corners_px` | ndarray (4×2) | Four corners of the opening |
| `lateral_error_norm` | float | Horizontal offset from frame centre, normalised [−1, 1] |
| `vertical_error_norm` | float | Vertical offset from frame centre, normalised [−1, 1] |
| `yaw_proxy` | float | Estimated yaw from post width asymmetry |
| `opening_width_px` | float | Opening width in pixels |
| `opening_height_px` | float | Opening height in pixels |
| `range_estimate_m` | float | Range from focal length + known gate width (if configured) |
| `pose_rvec` / `pose_tvec` | ndarray | 6-DoF pose from `solvePnP` (if camera calibrated) |
| `debug` | dict | All intermediate scores for diagnostics |

### Control loop usage

```python
if detection.is_valid:
    # Direct PID inputs — both in [-1, +1], zero = centred
    lateral_cmd = -detection.lateral_error_norm   # positive = gate is right → steer right
    vertical_cmd = -detection.vertical_error_norm  # positive = gate is above → climb
    yaw_cmd      =  detection.yaw_proxy            # positive = gate rotated clockwise

    # Throttle reduction as you approach (requires known gate width + focal length)
    if detection.range_estimate_m is not None:
        slow_down_factor = min(1.0, detection.range_estimate_m / 3.0)
```

---

## Configuration reference

All parameters live in `GateDetectorConfig` and can be overridden at construction:

```python
cfg = GateDetectorConfig(
    min_gate_confidence=0.50,       # validity threshold [0, 1]
    enable_multiscale=True,         # half-res supplemental pass
    enable_hough_verticals=True,    # Hough supplement for fragmented poles
    enable_adaptive_color_model=True, # histogram back-projection
    enable_adaptive_illuminant=True,  # grey-world hue shift
    enable_temporal_mask=False,     # EMA mask fusion (off by default)

    # Colour
    hsv_lower_blue=(85, 35, 18),
    hsv_upper_blue=(145, 255, 255),
    post_color_purity_hard_floor=0.08, # orange/warm veto floor

    # Geometry
    min_vertical_aspect=1.5,        # posts must be taller than wide
    max_post_aspect=14.0,           # thin rods / person legs
    max_opening_aspect=2.2,         # banner aspect kill
    min_gap_to_post_width=1.5,      # single-pole-split kill
    min_top_support_score=0.04,     # min checker bar evidence

    # Pose estimation (optional)
    gate_opening_width_m=1.4,
    gate_opening_height_m=1.4,
    fx_px=560.0,
    camera_matrix=K,                # 3×3 np.ndarray
    dist_coeffs=dist,               # 1×5 np.ndarray

    # Tracker
    lk_seed_min_consecutive=2,
    tracker_reset_missed_frames=15,
)
detector = EnhancedGateDetector(cfg)
```

---

## Known limitations and residual failure modes

| Scenario | Behaviour | Mitigation |
|---|---|---|
| Person wearing blue military/technical clothing | FP — blue clothing near a wall with shelves activates the colour mask and can form post-like pairs | Blue clothing with correct aspect and isolation is indistinguishable at single-frame level; temporal consistency (LK seed guard) reduces ghost rate |
| Gate at extreme yaw (>45°) | FN — one post heavily foreshortened, fails overlap filter | Perspective-aware aspect partially compensates; not fully solved |
| Uniform blue background (banner wall) | Handled — mask density gate and isolation veto | Confirmed working on competition footage |
| Orange/cream pillars | Handled — colour purity veto | Confirmed working |
| Wide blue banners | Handled — aspect veto + isolation | Confirmed working |
| Horizontal shelf/tarp edges | Handled — Hough column-density filter | Confirmed working |
| Square furniture/object blobs | Handled — `min_vertical_aspect=1.5` | Confirmed working |
| Single pole (one post visible) | No detection — correct behaviour | N/A |

---

## Annotated output

```python
annotated = detector.annotate(processed_frame, detection, mask)
cv2.imwrite("output.png", annotated)
```

The annotation overlays:
- **Green outer bbox** — valid detection; **orange** — sub-threshold
- **Cyan dashed inner bbox** — gate opening (fly-through target)
- **Yellow dots** — predicted opening corners
- **Red dot + line** — opening centre and bearing from frame centre
- **Mask thumbnail** (top-left quarter) — blue segmentation result
- **Status bar** — mode, confidence, lateral/vertical error, range estimate

---

## Resetting state between flights

```python
detector.reset_state()
```

Clears the LK tracker, temporal masks, colour model, and miss counter. Call between independent flight sequences to prevent state leakage.

---

## Dependencies

| Package | Version | Use |
|---|---|---|
| `opencv-python` | ≥ 4.5 | All CV operations |
| `numpy` | ≥ 1.20 | Array maths |

Python ≥ 3.9 required (dataclasses with `field`, `from __future__ import annotations`).

---

## File structure

```
gate_detector.py        # Complete self-contained implementation (2116 lines)
README.md               # This file
```

The file contains 8 classes and 46 methods. All public methods have docstrings explaining the reasoning, not just the interface.
