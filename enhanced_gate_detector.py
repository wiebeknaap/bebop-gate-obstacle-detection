"""
Enhanced Gate Detector — full pipeline rebuild
================================================
Novel additions on top of the original blue-first, geometry-aware detector:

  1. BlueColorModel        — online histogram back-projection (adaptive to
                              scene illuminant); replaces the fixed HSV band.
  2. TemporalMaskFusion    — exponential-decay accumulation of binary masks
                              across frames; suppresses transient noise and
                              fills short occlusion gaps at near-zero cost.
  3. LKGateTracker         — Lucas-Kanade sparse optical-flow tracker on the
                              four predicted gate corners; far more robust than
                              constant-velocity extrapolation during fast
                              manoeuvres or detection blackouts.
  4. Adaptive illuminant   — per-frame grey-world hue shift corrects warm/cool
                              artificial lighting without a calibration target.
  5. Hough vertical posts  — probabilistic Hough on the blue mask supplements
                              connected components; rescues fragmented poles.
  6. Multi-scale detection — independent half-resolution pass aggregates thin
                              far-field post pixels; best result wins.
  7. Soft geometric gates  — hard cutoffs replaced by continuous penalties;
                              angled / partially visible gates no longer vanish.
  8. Checkered top bar     — column-gradient alternation score detects the
                              black-white top bar more reliably than Canny
                              edge density.
  9. Shi-Tomasi corners    — corner-response map sampled at the four predicted
                              opening corners; real gate intersections light up.
 10. Gradient perpendicularity — Sobel-X alignment at post inner edges; a
                              vertical blue stripe has strongly horizontal
                              gradients at its boundaries.
 11. Horizontal scan profile — scanline at the opening mid-height confirms the
                              correct signature: post | gap | post.  Strongly
                              discriminates against blue walls / floors.
 12. Perspective-aware aspect — opening aspect target is corrected for the yaw
                              estimated from asymmetric post widths.
 13. Recalibrated weights   — all 12 terms re-balanced; w_profile added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GateDetectorConfig:
    rotate_portrait_frames: bool = True

    # ── Fixed HSV colour range (prior / fallback when model is cold) ──────
    # Lower hue raised from 85 → 95.
    # Vegetation in shade (grass, dark foliage, palm trunks) sits at H=75-94
    # (yellow-green to cyan-green range).  The TUDelft competition gate is
    # pure blue at H=100-120.  Raising the floor to 95 eliminates the
    # most common source of outdoor false positives while keeping all real gate
    # pixels (which also improves detection quality by giving a cleaner mask).
    hsv_lower_blue: tuple[int, int, int] = (95, 35, 18)
    hsv_upper_blue: tuple[int, int, int] = (145, 255, 255)
    min_bg_diff: int = 18
    min_br_diff: int = 12
    min_saturation_for_rgb_mask: int = 30
    blue_ratio_min: float = 0.34

    # ── Adaptive colour model (histogram back-projection) ─────────────────
    enable_adaptive_color_model: bool = True
    color_model_momentum: float = 0.88      # weight retained from previous model
    color_model_min_pixels: int = 120       # min mask pixels to trigger update
    backproject_weight: float = 0.55        # blend: backproject vs fixed-HSV mask

    # ── Adaptive illuminant (grey-world hue shift) ────────────────────────
    enable_adaptive_illuminant: bool = True
    max_hue_shift: int = 14                 # HSV hue units

    # ── Preprocessing ─────────────────────────────────────────────────────
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.2
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # ── Morphology ────────────────────────────────────────────────────────
    open_kernel_size: int = 3
    close_kernel_size: int = 5
    vertical_close_kernel_h: int = 11
    horizontal_close_kernel_w: int = 11

    # ── Temporal mask fusion ──────────────────────────────────────────────
    # Disabled by default: in scenes with persistent blue backgrounds (banners,
    # tarps, walls) the running average accumulates false blue that then
    # interferes with per-frame detection. Re-enable only in fully controlled
    # environments with no background blue.
    enable_temporal_mask: bool = False
    temporal_mask_alpha: float = 0.60       # weight for current-frame mask

    # ── Hough line verticals ──────────────────────────────────────────────
    enable_hough_verticals: bool = True
    hough_min_line_length_ratio: float = 0.07
    hough_max_line_gap_ratio: float = 0.05
    hough_angle_tolerance_deg: float = 22.0
    hough_cluster_x_tol: float = 14.0

    # ── Multi-scale (half-resolution supplemental pass) ───────────────────
    enable_multiscale: bool = True
    multiscale_confidence_advantage: float = 0.04  # half must exceed full by this

    # ── Connected-component filtering ─────────────────────────────────────
    min_component_area: int = 100
    # Raised from 0.9 → 1.5: square blobs (furniture, shelf corners, random objects)
    # have aspect ≈ 0.9–1.4. Real gate posts are always clearly taller than wide
    # (aspect ≥ 1.5) even at steep viewing angles.
    min_vertical_aspect: float = 1.5
    min_vertical_height_ratio: float = 0.05
    max_post_aspect: float = 14.0

    # ── Gate validity ─────────────────────────────────────────────────────
    min_gate_confidence: float = 0.50

    # ── Soft geometric thresholds (replace binary cutoffs) ────────────────
    soft_overlap_floor: float = 0.25
    soft_height_ratio_floor: float = 0.40
    soft_occupancy_ceiling: float = 0.30
    hard_occupancy_ceiling: float = 0.55

    # ── Hard aspect ratio cutoffs (banner / tarp suppression) ─────────────
    max_opening_aspect: float = 2.2
    max_outer_aspect:   float = 2.0

    # ── Minimum gap / post-width ratio (single-pole split suppression) ────
    # A pole that splits into two blobs has a tiny gap (< post width).
    # A real gate has opening_width >> post_width.
    # Require gap ≥ this factor × max(left_w, right_w).
    min_gap_to_post_width: float = 1.5

    # ── Checkered top bar minimum (most gate-specific feature) ────────────
    # Used as a score term only — NOT a hard gate. The mean_grad-based
    # normalization means real gate posts score 0.03-0.15; setting a hard
    # floor above 0.03 kills valid gates seen from distance.
    # The improved _score_checkered_top uses quantile normalization which
    # makes real gates score 0.15-0.60 and non-gates score 0.02-0.10.
    min_checker_score: float = 0.0   # disabled as hard gate; acts as score term only

    # ── Blue isolation — penalise when mask continues past posts ──────────
    isolation_check_width_ratio: float = 0.25
    isolation_hard_floor: float = 0.30
    isolation_soft_floor: float = 0.55

    # ── Color purity veto (false-positive suppression) ────────────────────
    post_hue_lo: int = 95   # matches hsv_lower_blue — vegetation (H<95) is not blue
    post_hue_hi: int = 155
    post_min_saturation: int = 28
    post_min_value: int = 20
    post_color_purity_hard_floor: float = 0.08
    post_color_purity_soft_floor: float = 0.22

    # ── Score weights ─────────────────────────────────────────────────────
    w_symmetry: float        = 0.14
    w_overlap: float         = 0.13
    w_gap: float             = 0.11
    w_aspect: float          = 0.09
    w_top_support: float     = 0.10
    w_hollow: float          = 0.06
    w_corner_response: float = 0.05
    w_gradient_perp: float   = 0.04
    w_profile: float         = 0.07
    w_color_purity: float    = 0.10
    w_isolation: float       = 0.08
    w_post_fill: float       = 0.02
    w_balance: float         = 0.01
    w_center: float          = 0.01

    # ── Sub-threshold filters (applied before scoring) ────────────────────
    min_opening_width_ratio: float = 0.07
    min_opening_height_ratio: float = 0.09
    min_top_support_score: float = 0.04   # lowered — only kills zero-evidence cases
    min_hollow_score: float = 0.08

    # ── Global mask density gate ──────────────────────────────────────────
    # Only kills truly flooded frames (whole wall is blue). Real gate + blue
    # background: ~6-10%. Full tarp: ~25-40%.
    max_mask_density: float = 0.28        # raised from 0.14

    # ── Temporal mask spatial dispersion check ────────────────────────────
    # Raised — a single post seen alone is one blob; don't kill that case.
    max_single_blob_fraction: float = 0.90  # only kills nearly-single-blob

    # ── Consecutive-miss tracker reset ───────────────────────────────────
    tracker_reset_missed_frames: int = 15   # raised — don't reset during brief occlusions

    # ── LK tracker seeding guard ──────────────────────────────────────────
    lk_seed_min_consecutive: int = 2

    # ── Ring hypothesis ───────────────────────────────────────────────────
    enable_ring_hypothesis: bool = False
    strong_ring_confidence: float = 0.76
    ring_selection_margin: float = 0.12

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker_measurement_confidence: float = 0.26

    # ── Pose estimation (optional) ────────────────────────────────────────
    gate_opening_width_m: Optional[float] = None
    gate_opening_height_m: Optional[float] = None
    fx_px: Optional[float] = None
    fy_px: Optional[float] = None
    cx_px: Optional[float] = None
    cy_px: Optional[float] = None
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GateDetection:
    is_valid: bool
    confidence: float
    mode: str
    image_size: tuple[int, int]
    center_px: Optional[tuple[float, float]] = None
    outer_bbox: Optional[tuple[int, int, int, int]] = None
    opening_bbox: Optional[tuple[int, int, int, int]] = None
    corners_px: Optional[np.ndarray] = None
    lateral_error_norm: Optional[float] = None
    vertical_error_norm: Optional[float] = None
    yaw_proxy: Optional[float] = None
    opening_width_px: Optional[float] = None
    opening_height_px: Optional[float] = None
    range_estimate_m: Optional[float] = None
    pose_rvec: Optional[np.ndarray] = None
    pose_tvec: Optional[np.ndarray] = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackerState:
    """Kept for backwards compatibility with AlphaBetaGateTracker."""
    valid: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Novel component 1 — Online adaptive colour model
# ---------------------------------------------------------------------------

class BlueColorModel:
    """Online-learning HSV histogram back-projection model.

    Why this beats fixed HSV thresholds
    ------------------------------------
    Competition lighting shifts the perceived hue of blue gate poles by up to
    ±10 HSV units.  A fixed band either misses pixels (band too narrow) or
    admits noise (band too wide).  This model maintains a 2-D histogram over
    (Hue, Saturation) — Value is excluded because it is strongly correlated
    with illuminant intensity rather than object identity.  The histogram is
    initialised from the config's range and refined on every confident
    detection, converging to the actual appearance of this particular gate
    under these particular lights.

    Usage: call backproject() to get a 0-255 saliency map; call update()
    after each confident detection to refine the model.
    """

    H_BINS = 64   # 0-179 → 64 bins ≈ 2.8° per bin
    S_BINS = 32   # 0-255 → 32 bins ≈ 8 levels per bin

    def __init__(self, config: GateDetectorConfig) -> None:
        self._config = config
        self._hist: Optional[np.ndarray] = None
        self._update_count: int = 0
        self._init_from_config()

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def update_count(self) -> int:
        return self._update_count

    # Blue hue range in the H_BINS space (0-179 → 0-63 bins)
    _HUE_LO_BIN: int = int(95  * 64 / 180)   # ≈ bin 34  (matches hsv_lower_blue)
    _HUE_HI_BIN: int = int(155 * 64 / 180)   # ≈ bin 55

    def update(self, frame_hsv: np.ndarray, mask: np.ndarray) -> None:
        """Blend gate-pixel histogram from this frame into the running model.

        Poisoning guard: before blending, verify that the pixels under the
        mask are actually blue by checking the weighted mean hue.  Orange/cream
        pixels (hue 0-30) would pull the histogram away from blue, causing
        subsequent back-projection to fire on non-gate objects.
        """
        n_px = int(mask.sum()) // 255
        if n_px < self._config.color_model_min_pixels:
            return

        # ── Poisoning guard: sample mean hue of masked pixels ────────────
        hue_channel = frame_hsv[:, :, 0]
        masked_hues = hue_channel[mask > 0].astype(np.float32)
        if len(masked_hues) < 10:
            return
        median_hue = float(np.median(masked_hues))
        if not (self._config.post_hue_lo <= median_hue <= self._config.post_hue_hi):
            # Masked pixels are not blue — do not update model
            return

        new_hist = cv2.calcHist(
            [frame_hsv], [0, 1], mask,
            [self.H_BINS, self.S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(new_hist, new_hist, 0, 255, cv2.NORM_MINMAX)
        if self._hist is None:
            self._hist = new_hist
        else:
            m = self._config.color_model_momentum
            cv2.addWeighted(self._hist, m, new_hist, 1.0 - m, 0, self._hist)

        # ── Hue centroid drift guard: if model has drifted outside blue, reset
        if self._hist is not None and self._hue_centroid_out_of_range():
            self._init_from_config()
            return

        self._update_count += 1

    def _hue_centroid_out_of_range(self) -> bool:
        """Return True if the histogram's hue centre-of-mass has drifted
        outside the expected blue hue band — indicating model poisoning."""
        if self._hist is None:
            return False
        h_marginal = self._hist.sum(axis=1).ravel().astype(np.float32)
        total = float(h_marginal.sum())
        if total < 1.0:
            return False
        h_bins = np.arange(self.H_BINS, dtype=np.float32)
        centroid_bin = float((h_marginal * h_bins).sum() / total)
        return not (self._HUE_LO_BIN <= centroid_bin <= self._HUE_HI_BIN)

    def backproject(self, frame_hsv: np.ndarray) -> np.ndarray:
        """Return a saliency map (uint8, 0-255) via histogram back-projection."""
        if self._hist is None:
            return np.zeros(frame_hsv.shape[:2], dtype=np.uint8)
        disc = cv2.calcBackProject(
            [frame_hsv], [0, 1], self._hist, [0, 180, 0, 256], scale=1.0,
        )
        disc = cv2.GaussianBlur(disc, (5, 5), 0)
        return disc

    # ── private ─────────────────────────────────────────────────────────────

    def _init_from_config(self) -> None:
        lo = self._config.hsv_lower_blue
        hi = self._config.hsv_upper_blue
        h_vals = np.linspace(lo[0], hi[0], 50).astype(np.uint8)
        s_vals = np.linspace(max(lo[1], 40), 255, 30).astype(np.uint8)
        v_vals = np.linspace(max(lo[2], 30), 255, 10).astype(np.uint8)
        H, S, V = np.meshgrid(h_vals, s_vals, v_vals)
        synthetic = (
            np.stack([H.ravel(), S.ravel(), V.ravel()], axis=1)
            .reshape(-1, 1, 3)
            .astype(np.uint8)
        )
        self._hist = cv2.calcHist(
            [synthetic], [0, 1], None,
            [self.H_BINS, self.S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(self._hist, self._hist, 0, 255, cv2.NORM_MINMAX)


# ---------------------------------------------------------------------------
# Novel component 2 — Temporal mask fusion
# ---------------------------------------------------------------------------

class TemporalMaskFusion:
    """Exponential-decay temporal accumulation of binary segmentation masks.

    A genuine gate pixel is consistently blue across frames.  A false positive
    from floor glare or clothing fires only once or twice.  By maintaining a
    running weighted average of the binary mask, consistent pixels accumulate
    above the 50% threshold while transient noise decays below it.

    Computation cost: one multiply-add per pixel per frame — essentially free
    on any platform.
    """

    def __init__(self, alpha: float = 0.60) -> None:
        """
        Args:
            alpha: weight for the *current* frame (0.55–0.70 recommended).
                   Higher → faster response; lower → stronger noise rejection.
        """
        self.alpha = float(alpha)
        self._running: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._running = None

    def update(self, mask: np.ndarray) -> np.ndarray:
        """Blend `mask` into the running average; return thresholded mask."""
        m = mask.astype(np.float32)
        if self._running is None or self._running.shape != m.shape:
            self._running = m.copy()
        else:
            self._running = self.alpha * m + (1.0 - self.alpha) * self._running
        return (self._running > 127.0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Tracker — legacy alpha-beta (kept for API compatibility)
# ---------------------------------------------------------------------------

class AlphaBetaGateTracker:
    """Constant-velocity tracker.  Kept for backwards compatibility.
    The active tracker in EnhancedGateDetector is now LKGateTracker.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 0.22,
        confidence_decay: float = 0.92,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.confidence_decay = float(confidence_decay)
        self.state = TrackerState()

    def reset(self) -> None:
        self.state = TrackerState()

    def predict(self, dt: float) -> TrackerState:
        if not self.state.valid:
            return self.state
        self.state.center_x += self.state.vx * dt
        self.state.center_y += self.state.vy * dt
        self.state.width += self.state.vw * dt
        self.state.height += self.state.vh * dt
        self.state.confidence *= self.confidence_decay
        return self.state

    def update(
        self,
        detection: GateDetection,
        dt: float = 1 / 30.0,
        accept_confidence: float = 0.0,
    ) -> TrackerState:
        has_measurement = detection.opening_bbox is not None and (
            detection.is_valid or detection.confidence >= accept_confidence
        )
        if not self.state.valid:
            if has_measurement:
                x, y, w, h = detection.opening_bbox  # type: ignore[misc]
                self.state = TrackerState(
                    valid=True,
                    center_x=x + 0.5 * w,
                    center_y=y + 0.5 * h,
                    width=float(w),
                    height=float(h),
                    confidence=float(detection.confidence),
                )
            return self.state
        self.predict(dt)
        if not has_measurement:
            return self.state
        x, y, w, h = detection.opening_bbox  # type: ignore[misc]
        zx, zy, zw, zh = x + 0.5 * w, y + 0.5 * h, float(w), float(h)
        rx = zx - self.state.center_x
        ry = zy - self.state.center_y
        rw = zw - self.state.width
        rh = zh - self.state.height
        self.state.center_x += self.alpha * rx
        self.state.center_y += self.alpha * ry
        self.state.width += self.alpha * rw
        self.state.height += self.alpha * rh
        if dt > 1e-6:
            self.state.vx += (self.beta / dt) * rx
            self.state.vy += (self.beta / dt) * ry
            self.state.vw += (self.beta / dt) * rw
            self.state.vh += (self.beta / dt) * rh
        self.state.confidence = max(
            self.state.confidence * self.confidence_decay,
            float(detection.confidence),
        )
        return self.state


# ---------------------------------------------------------------------------
# Novel component 3 — Lucas-Kanade optical flow tracker
# ---------------------------------------------------------------------------

class LKGateTracker:
    """Sparse Lucas-Kanade optical flow tracker on gate opening corners.

    Why LK beats constant-velocity extrapolation
    --------------------------------------------
    Alpha-beta trackers extrapolate from a velocity estimate that was valid
    a few frames ago.  During fast roll / yaw manoeuvres the gate can move
    30–50 px in a single frame — far beyond what a linear model predicts.
    LK propagates the *actual image structure* (gradient patches around each
    corner) through the pyramid, so it stays locked even at high velocity.

    Key properties:
    - 4 tracked points, 3-level pyramid, 21×21 window.
    - Confidence decays at `confidence_decay` per frame; falls back gracefully.
    - Seamlessly reseeded whenever a fresh detection is available.
    - Cost: ~0.3 ms on ARM Cortex-A57 (4 pts, level 3).
    """

    _LK = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    def __init__(self, confidence_decay: float = 0.93) -> None:
        self.confidence_decay = float(confidence_decay)
        self._corners: Optional[np.ndarray] = None   # (4, 1, 2) float32
        self._prev_gray: Optional[np.ndarray] = None
        self._confidence: float = 0.0

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def valid(self) -> bool:
        return self._corners is not None and self._confidence > 0.01

    def reset(self) -> None:
        self._corners = None
        self._prev_gray = None
        self._confidence = 0.0

    def update_from_detection(self, det: GateDetection, gray: np.ndarray) -> None:
        """Reseed the tracker from a fresh detection."""
        if det.corners_px is not None:
            self._corners = det.corners_px.reshape(4, 1, 2).astype(np.float32)
        elif det.opening_bbox is not None:
            x, y, w, h = det.opening_bbox
            self._corners = np.array([
                [[float(x),     float(y)]],
                [[float(x + w), float(y)]],
                [[float(x + w), float(y + h)]],
                [[float(x),     float(y + h)]],
            ], dtype=np.float32)
        else:
            return
        self._prev_gray = gray.copy()
        self._confidence = float(det.confidence)

    def predict(self, gray: np.ndarray, image_size: tuple[int, int]) -> Optional[np.ndarray]:
        """Track corners into the current frame.  Returns (4,1,2) array or None."""
        if self._corners is None or self._prev_gray is None:
            self._prev_gray = gray.copy()
            return None

        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._corners, None, **self._LK
        )
        self._prev_gray = gray.copy()

        if new_corners is None or status is None:
            self._confidence *= self.confidence_decay ** 3
            return None

        good = status.ravel() == 1
        iw, ih = image_size
        for idx in range(4):
            if good[idx]:
                cx, cy = new_corners[idx, 0]
                if cx < 0 or cx >= iw or cy < 0 or cy >= ih:
                    good[idx] = False

        if good.sum() < 2:
            self._confidence *= self.confidence_decay ** 2
            self._corners = None
            return None

        self._corners = new_corners
        self._confidence *= self.confidence_decay
        return new_corners

    def to_detection(
        self,
        image_size: tuple[int, int],
        min_confidence: float,
    ) -> Optional[GateDetection]:
        """Convert current tracked state to a GateDetection (is_valid=False)."""
        if self._corners is None or self._confidence < min_confidence * 0.5:
            return None

        pts = self._corners.reshape(4, 2)
        x1 = float(pts[:, 0].min())
        x2 = float(pts[:, 0].max())
        y1 = float(pts[:, 1].min())
        y2 = float(pts[:, 1].max())

        iw, ih = image_size
        bx = max(0, min(iw - 1, int(round(x1))))
        by = max(0, min(ih - 1, int(round(y1))))
        bw = max(1, min(iw - bx, int(round(x2 - x1))))
        bh = max(1, min(ih - by, int(round(y2 - y1))))

        det = GateDetection(
            is_valid=False,
            confidence=float(min(self._confidence, min_confidence - 1e-3)),
            mode="lk_tracked",
            image_size=image_size,
            center_px=(float(bx + 0.5 * bw), float(by + 0.5 * bh)),
            outer_bbox=(bx, by, bw, bh),
            opening_bbox=(bx, by, bw, bh),
            corners_px=pts.astype(np.float32),
            opening_width_px=float(bw),
            opening_height_px=float(bh),
            debug={"source": "lk_tracker", "lk_confidence": float(self._confidence)},
        )
        return det


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class EnhancedGateDetector:
    """Blue-first, geometry-aware, temporally-robust gate detector.

    Pipeline overview (per frame)
    ------------------------------
    preprocess → adaptive-HSV-segment → temporal-fuse → [ring | post-pair]
    → [half-scale post-pair] → select best → LK-track → pose estimate → output
    """

    def __init__(self, config: Optional[GateDetectorConfig] = None) -> None:
        self.config = config or GateDetectorConfig()
        # Novel components
        self._color_model = BlueColorModel(self.config)
        self._temporal_mask = TemporalMaskFusion(self.config.temporal_mask_alpha)
        self._temporal_mask_half = TemporalMaskFusion(self.config.temporal_mask_alpha)
        self._lk_tracker = LKGateTracker(confidence_decay=0.93)
        # Legacy tracker (API compatibility)
        self.tracker = AlphaBetaGateTracker(alpha=0.70, beta=0.22, confidence_decay=0.92)
        # State for temporal stability guards
        self._consecutive_misses: int = 0       # frames since last valid detection
        self._consecutive_valids: int = 0       # consecutive valid detections seen

    def reset_state(self) -> None:
        """Full state reset — call when starting a new flight or sequence."""
        self._lk_tracker.reset()
        self._temporal_mask.reset()
        self._temporal_mask_half.reset()
        self._color_model._init_from_config()
        self._color_model._update_count = 0
        self.tracker.reset()
        self._consecutive_misses = 0
        self._consecutive_valids = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = frame_bgr.copy()
        if self.config.rotate_portrait_frames and frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.config.enable_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid,
            )
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame

    def segment_gate_blue(self, frame_bgr: np.ndarray, hue_shift: int = 0) -> np.ndarray:
        """Segment blue gate pixels.  Combines fixed-HSV with back-projection."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(frame_bgr)
        b_i = b.astype(np.int16)
        g_i = g.astype(np.int16)
        r_i = r.astype(np.int16)
        sum_rgb = b.astype(np.float32) + g.astype(np.float32) + r.astype(np.float32) + 1.0
        blue_ratio = b.astype(np.float32) / sum_rgb

        # Illuminant-corrected HSV band
        lo = self.config.hsv_lower_blue
        hi = self.config.hsv_upper_blue
        lower = np.array([
            int(np.clip(lo[0] + hue_shift, 0, 179)), lo[1], lo[2],
        ], dtype=np.uint8)
        upper = np.array([
            int(np.clip(hi[0] + hue_shift, 0, 179)), hi[1], hi[2],
        ], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower, upper)

        # Back-projection blend (only when model has been updated at least once)
        if self.config.enable_adaptive_color_model and self._color_model.update_count > 0:
            bp = self._color_model.backproject(hsv)
            # Force backproject output to stay within a wide but orange-excluding
            # hue range.  This prevents a mildly drifted model from segmenting
            # orange / cream objects even when the back-projection fires on them.
            bp_hue_gate = cv2.inRange(
                hsv,
                np.array([self.config.post_hue_lo, self.config.post_min_saturation, self.config.post_min_value], dtype=np.uint8),
                np.array([self.config.post_hue_hi, 255, 255], dtype=np.uint8),
            )
            mask_bp = cv2.bitwise_and(
                cv2.bitwise_and((bp > 60).astype(np.uint8) * 255, bp_hue_gate),
                (hsv[:, :, 1] > self.config.min_saturation_for_rgb_mask).astype(np.uint8) * 255,
            )
            w = self.config.backproject_weight
            combined = w * mask_bp.astype(np.float32) + (1.0 - w) * mask_hsv.astype(np.float32)
            mask_hsv = (combined > 127.0).astype(np.uint8) * 255

        # RGB ratio gate
        mask_rgb = (
            (b_i - g_i > self.config.min_bg_diff)
            & (b_i - r_i > self.config.min_br_diff)
            & (hsv[:, :, 1] > self.config.min_saturation_for_rgb_mask)
        ).astype(np.uint8) * 255
        mask_ratio = (blue_ratio > self.config.blue_ratio_min).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask_hsv, cv2.bitwise_or(mask_rgb, mask_ratio))

        # Morphological refinement
        k_open  = np.ones((self.config.open_kernel_size,  self.config.open_kernel_size),  np.uint8)
        k_close = np.ones((self.config.close_kernel_size, self.config.close_kernel_size), np.uint8)
        k_vc    = np.ones((self.config.vertical_close_kernel_h, 3), np.uint8)
        k_hc    = np.ones((3, self.config.horizontal_close_kernel_w), np.uint8)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_vc)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_hc)

        # Suppress bottom strip (floor-mat clutter)
        h, _ = mask.shape
        mask[int(0.95 * h):, :] = 0
        return mask

    def _check_mask_density(self, mask: np.ndarray) -> bool:
        """Return True if mask is safe to run hypotheses on.

        Global density gate
        -------------------
        A large blue banner/tarp fills 15-40% of the frame with blue.
        Real gate posts fill 1-5%.  If the mask exceeds `max_mask_density`,
        skip hypothesis testing entirely — there is no isolated post pair.
        Also protects against the temporal mask accumulating banner blue into
        a state where it looks like there's always a gate present.
        """
        total_px = mask.shape[0] * mask.shape[1]
        blue_px  = int(mask.sum()) // 255
        density  = float(blue_px) / max(total_px, 1)
        return density <= self.config.max_mask_density

    def _check_mask_dispersion(self, mask: np.ndarray) -> bool:
        """Return True if the blue blobs are compact enough to be posts.

        Temporal mask dispersion check
        --------------------------------
        The temporal mask accumulates pixels that are *consistently blue*
        across frames.  A banner in the background is also consistently blue,
        so after a few seconds the banner region saturates in the running
        average and looks like a genuine detection.

        However, a banner forms one huge connected blob, while gate posts
        form exactly two compact blobs.  If the largest connected component
        contains more than `max_single_blob_fraction` of all blue pixels,
        the frame is dominated by a single large blue region — reject it.
        """
        blue_px = int(mask.sum()) // 255
        if blue_px < 50:
            return True   # too few pixels to judge; allow through

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n < 2:
            return False  # single component covering everything → banner

        # stats[0] is background; find largest foreground blob
        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
        largest = max(areas)
        fraction = largest / max(blue_px, 1)
        return fraction <= self.config.max_single_blob_fraction

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, GateDetection]:
        frame = self.preprocess(frame_bgr)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_h, image_w = frame.shape[:2]

        def _no_gate(reason: str) -> tuple[np.ndarray, np.ndarray, GateDetection]:
            """Shared path for all negative outcomes — updates miss counter."""
            self._consecutive_misses += 1
            self._consecutive_valids = 0
            # Hard reset of all temporal state after enough consecutive misses
            if self._consecutive_misses >= self.config.tracker_reset_missed_frames:
                self._lk_tracker.reset()
                self._temporal_mask.reset()
                self._temporal_mask_half.reset()
                self._consecutive_misses = 0
            return frame, mask_out, GateDetection(
                is_valid=False, confidence=0.0, mode="none",
                image_size=(image_w, image_h),
                debug={"reason": reason, "missed_frames": self._consecutive_misses},
            )

        # 1. Advance LK tracker into this frame immediately
        lk_corners = self._lk_tracker.predict(gray, (image_w, image_h))

        # 2. Adaptive illuminant hue correction
        hue_shift = (
            self._adaptive_illuminant_hue_shift(frame)
            if self.config.enable_adaptive_illuminant else 0
        )

        # 3. Colour segmentation
        raw_mask = self.segment_gate_blue(frame, hue_shift)

        # 4. Global mask density gate — abort early if frame is flooded with blue
        if not self._check_mask_density(raw_mask):
            mask_out = raw_mask
            # Also reset temporal mask so it doesn't accumulate banner blue
            if self.config.enable_temporal_mask:
                self._temporal_mask.reset()
                self._temporal_mask_half.reset()
            self._consecutive_misses += 1
            self._consecutive_valids = 0
            if self._consecutive_misses >= self.config.tracker_reset_missed_frames:
                self._lk_tracker.reset()
                self._consecutive_misses = 0
            return frame, mask_out, GateDetection(
                is_valid=False, confidence=0.0, mode="none",
                image_size=(image_w, image_h),
                debug={"reason": "mask_density_exceeded"},
            )

        # 5. Apply temporal fusion
        if self.config.enable_temporal_mask:
            mask = self._temporal_mask.update(raw_mask)
        else:
            mask = raw_mask
        mask_out = mask

        # 6. Skip mask dispersion check — the Hough column-density filter
        # already rejects posts fabricated from wide blobs. The dispersion
        # check was incorrectly killing close-range gate views where the
        # gate's own blue structural element is a single large blob.

        # 7. Derived maps (shared across scoring methods)
        frame_hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.Canny(gray, 60, 140)

        # 8. Hypotheses (full scale)
        ring_candidate = None
        if self.config.enable_ring_hypothesis:
            ring_candidate = self._detect_ring_hypothesis(mask, image_w, image_h)

        post_candidate = self._detect_post_pair_hypothesis(
            mask, edges, gray, gx, gy, frame_hsv_full, image_w, image_h
        )

        # 9. Multi-scale (half resolution) — helps at distance
        if self.config.enable_multiscale and image_w >= 64 and image_h >= 64:
            hw, hh = image_w // 2, image_h // 2
            frame_h  = cv2.resize(frame, (hw, hh))
            gray_h   = cv2.resize(gray,  (hw, hh))
            raw_mask_h = self.segment_gate_blue(frame_h, hue_shift)
            if self.config.enable_temporal_mask and self._check_mask_density(raw_mask_h):
                mask_hr = self._temporal_mask_half.update(raw_mask_h)
            else:
                mask_hr = raw_mask_h
            gx_h    = cv2.Sobel(gray_h, cv2.CV_32F, 1, 0, ksize=3)
            gy_h    = cv2.Sobel(gray_h, cv2.CV_32F, 0, 1, ksize=3)
            edges_h = cv2.Canny(gray_h, 60, 140)
            frame_hsv_half = cv2.cvtColor(frame_h, cv2.COLOR_BGR2HSV)
            half_raw = self._detect_post_pair_hypothesis(
                mask_hr, edges_h, gray_h, gx_h, gy_h, frame_hsv_half, hw, hh
            )
            if half_raw is not None:
                half_scaled = self._scale_detection(half_raw, 2.0)
                adv = self.config.multiscale_confidence_advantage
                if post_candidate is None or half_scaled.confidence > post_candidate.confidence + adv:
                    post_candidate = half_scaled

        # 10. Select best hypothesis
        selected = self._select_candidate(ring_candidate, post_candidate)

        if selected is not None:
            selected = self._add_control_quantities(selected)

            # Update adaptive colour model with gate pixels from this frame
            if self.config.enable_adaptive_color_model and selected.confidence >= 0.42:
                self._color_model.update(frame_hsv_full, mask)

            # LK tracker seeding guard: only seed after K consecutive valid detections.
            # Prevents a single-frame false positive from generating a ghost track.
            if selected.is_valid:
                self._consecutive_valids += 1
                self._consecutive_misses = 0
                if self._consecutive_valids >= self.config.lk_seed_min_consecutive:
                    self._lk_tracker.update_from_detection(selected, gray)
            else:
                # Sub-threshold candidate — don't count as a valid, don't seed LK
                self._consecutive_valids = max(0, self._consecutive_valids - 1)
                self._consecutive_misses += 1

            # Legacy alpha-beta tracker always updates
            self.tracker.update(selected, accept_confidence=self.config.tracker_measurement_confidence)

            if self._can_estimate_pose(selected):
                self._estimate_pose(selected)

            return frame, mask, selected

        # 11. Fallback: LK tracker — only if it was seeded from confirmed detections
        if lk_corners is not None and self._consecutive_misses < self.config.tracker_reset_missed_frames:
            tracked = self._lk_tracker.to_detection(
                (image_w, image_h), self.config.min_gate_confidence
            )
            if tracked is not None:
                self._consecutive_misses += 1
                tracked = self._add_control_quantities(tracked)
                return frame, mask, tracked

        return _no_gate("no hypothesis survived")

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detection: GateDetection,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        if mask is not None:
            ms = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
            ms = cv2.cvtColor(ms, cv2.COLOR_GRAY2BGR)
            vis[: ms.shape[0], : ms.shape[1]] = ms

        if detection.outer_bbox is not None:
            x, y, bw, bh = detection.outer_bbox
            color = (0, 200, 0) if detection.is_valid else (0, 120, 255)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

        if detection.opening_bbox is not None:
            x, y, bw, bh = detection.opening_bbox
            color = (0, 255, 255) if detection.is_valid else (0, 165, 255)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

        if detection.corners_px is not None:
            corners = detection.corners_px.astype(int).reshape(-1, 1, 2)
            cv2.polylines(vis, [corners], True, (255, 255, 0), 2)

        if detection.center_px is not None:
            cx, cy = map(int, detection.center_px)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, h // 2), (cx, cy), (0, 0, 255), 1)

        mode_color = {
            "post_pair": (0, 255, 0),
            "ring":      (255, 200, 0),
            "lk_tracked": (200, 200, 0),
            "none":      (80, 80, 80),
        }.get(detection.mode, (180, 180, 180))

        status = f"{detection.mode}  conf={detection.confidence:.2f}"
        if detection.lateral_error_norm is not None and detection.vertical_error_norm is not None:
            status += f"  ex={detection.lateral_error_norm:+.2f} ey={detection.vertical_error_norm:+.2f}"
        if detection.range_estimate_m is not None:
            status += f"  rng~{detection.range_estimate_m:.2f}m"
        if self._color_model.update_count > 0:
            status += f"  cm={self._color_model.update_count}"
        cv2.putText(vis, status, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_color, 2)
        return vis

    # ── Novel scoring helpers ────────────────────────────────────────────────

    def _adaptive_illuminant_hue_shift(self, frame_bgr: np.ndarray) -> int:
        """Grey-world hue-shift correction.

        If the scene has a warm (reddish/yellowish) cast, the perceived blue
        of the gate shifts toward cyan (higher HSV hue).  We estimate the
        illuminant colour temperature from channel means and compensate.
        """
        b_mean = float(frame_bgr[:, :, 0].mean())
        g_mean = float(frame_bgr[:, :, 1].mean())
        r_mean = float(frame_bgr[:, :, 2].mean())
        scene_mean = (b_mean + g_mean + r_mean) / 3.0 + 1.0
        # Positive blue excess → scene is already bluish → shift hue slightly down
        # Negative blue excess → warm scene → perceived blue is shifted up
        hue_shift = int(np.clip(
            4.0 * (b_mean - scene_mean) / scene_mean,
            -self.config.max_hue_shift,
            self.config.max_hue_shift,
        ))
        return hue_shift

    def _hough_vertical_segments(
        self, mask: np.ndarray, image_h: int, image_w: int
    ) -> list[dict[str, Any]]:
        """Extract blue vertical post candidates via probabilistic Hough.

        Advantage over connected components
        ------------------------------------
        A pole in partial shadow (or seen through thin mesh) fragments into
        2-3 sub-threshold CC blobs.  HoughLinesP finds line *segments* even
        in a fragmented mask; clustering by x-position then reconstructs the
        full post span.

        Column-density filter (replaces earlier wide-blob veto)
        --------------------------------------------------------
        Hough can find vertical *edges* of wide horizontal blobs (shelves,
        tarps, clothing).  These fabricated posts are distinguishable from
        real fragmented posts because at their x-position the blue pixels
        are spread across the horizontal extent of the blob, not concentrated
        in a narrow vertical column.

        We measure the fraction of rows in a 3-pixel-wide window at each
        Hough cluster's cx that contain at least one blue pixel.  Real posts
        have a high fraction (blue runs vertically); shelf-edge artefacts
        have a low fraction (blue runs horizontally away from that column).
        """
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        dilated = cv2.dilate(mask, k, iterations=1)

        min_len = max(12, int(self.config.hough_min_line_length_ratio * image_h))
        max_gap = max(8,  int(self.config.hough_max_line_gap_ratio   * image_h))

        lines = cv2.HoughLinesP(
            dilated, rho=1, theta=np.pi / 180,
            threshold=25,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )
        if lines is None:
            return []

        tol = self.config.hough_angle_tolerance_deg
        segments: list[dict] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dy < 1:
                continue
            if np.degrees(np.arctan2(dx, dy)) > tol:
                continue
            segments.append({
                "cx": (x1 + x2) / 2.0,
                "y_top": float(min(y1, y2)),
                "y_bot": float(max(y1, y2)),
                "length": float(np.hypot(dx, dy)),
            })

        if not segments:
            return []

        segments.sort(key=lambda s: s["cx"])
        x_tol = self.config.hough_cluster_x_tol
        clusters: list[dict] = []
        for seg in segments:
            if clusters and abs(seg["cx"] - clusters[-1]["_cx"]) < x_tol:
                clusters[-1]["segs"].append(seg)
                clusters[-1]["_cx"] = float(np.mean([s["cx"] for s in clusters[-1]["segs"]]))
            else:
                clusters.append({"segs": [seg], "_cx": seg["cx"]})

        posts: list[dict] = []
        for cl in clusters:
            segs = cl["segs"]
            total_len = sum(s["length"] for s in segs)
            if total_len < 0.06 * image_h:
                continue
            y_top = float(min(s["y_top"] for s in segs))
            y_bot = float(max(s["y_bot"] for s in segs))
            cx    = float(cl["_cx"])

            # Column-density filter: at this x-position, does blue actually
            # run vertically?  Measure the fraction of rows in the cluster's
            # span that have at least one blue pixel in a 5-wide window.
            cx_int = int(round(cx))
            x_lo = max(0, cx_int - 2)
            x_hi = min(image_w, cx_int + 3)
            y_lo = max(0, int(y_top))
            y_hi = min(image_h, int(y_bot) + 1)
            if y_hi > y_lo and x_hi > x_lo:
                col_strip = mask[y_lo:y_hi, x_lo:x_hi]
                row_has_blue = (col_strip.max(axis=1) > 0).sum()
                span = y_hi - y_lo
                col_fraction = row_has_blue / max(span, 1)
            else:
                col_fraction = 0.0

            # Threshold: real posts have col_fraction ≥ 0.18 (18% of vertical
            # span has blue).  Shelf/tarp edges have col_fraction ≈ 0.05-0.12
            # because the blob is horizontal, not at this narrow column.
            if col_fraction < 0.18:
                continue

            h_seg = max(1.0, y_bot - y_top)
            posts.append({
                "x":   int(round(cx - 5)),
                "y":   int(round(y_top)),
                "w":   10,
                "h":   int(round(h_seg)),
                "area": int(total_len * 8),
                "cx":  cx,
                "cy":  float((y_top + y_bot) / 2.0),
                "aspect": float(h_seg / 10.0),
                "fill": float(col_fraction),   # use measured fill instead of hardcoded 0.5
                "source": "hough",
            })
        return posts

    def _score_post_color_purity(
        self,
        frame_hsv: np.ndarray,
        post: dict[str, Any],
    ) -> float:
        """Measure what fraction of the candidate post's pixels are genuinely blue.

        This is the primary false-positive suppressor for orange / cream / warm-
        colored pillars.  It operates on RAW HSV pixel values inside the post
        bounding box — completely independent of the segmentation mask and the
        adaptive colour model — so it cannot be fooled by model drift or mask
        pollution.

        Why orange pillars pass the geometric filters
        ---------------------------------------------
        Orange and cream pillars are vertical, tall, and often appear in pairs
        flanking a space.  They score well on symmetry, overlap, gap ratio, and
        profile.  The only reliable discriminator that doesn't require a gate to
        already be detected is the raw pixel color.

        Implementation
        ---------------
        1. Extract the HSV patch inside the post bounding box.
        2. Build a validity mask: ignore near-grey (S < threshold) and near-black
           (V < threshold) pixels — they provide no hue information.
        3. Of the remaining chromatic pixels, count the fraction with H ∈ [hue_lo,
           hue_hi].  This is the purity score.

        Orange pillars: purity ≈ 0.02–0.08 (orange H ≈ 10–25, far outside blue)
        Blue posts:     purity ≈ 0.65–0.95
        """
        cfg = self.config
        img_h, img_w = frame_hsv.shape[:2]

        x  = int(np.clip(post["x"], 0, img_w - 1))
        y  = int(np.clip(post["y"], 0, img_h - 1))
        x2 = int(np.clip(post["x"] + post["w"], x + 1, img_w))
        y2 = int(np.clip(post["y"] + post["h"], y + 1, img_h))

        patch = frame_hsv[y:y2, x:x2]
        if patch.size == 0:
            return 0.0

        H = patch[:, :, 0].ravel().astype(np.int16)
        S = patch[:, :, 1].ravel().astype(np.int16)
        V = patch[:, :, 2].ravel().astype(np.int16)

        chromatic = (S >= cfg.post_min_saturation) & (V >= cfg.post_min_value)
        n_chromatic = int(chromatic.sum())
        if n_chromatic < 4:
            # Post patch has no chromatic pixels — could be dark / shadowed;
            # don't veto, but return a neutral mid score so the pair isn't
            # heavily penalised for being in shadow.
            return 0.35

        H_chrom = H[chromatic]
        n_blue = int(((H_chrom >= cfg.post_hue_lo) & (H_chrom <= cfg.post_hue_hi)).sum())
        purity = float(n_blue) / float(n_chromatic)
        return float(np.clip(purity, 0.0, 1.0))

    def _score_blue_isolation(
        self,
        mask: np.ndarray,
        left: dict[str, Any],
        right: dict[str, Any],
        image_w: int,
        image_h: int,
    ) -> float:
        """Score how isolated the posts are from surrounding blue.

        The central diagnostic for blue-banner false positives
        -------------------------------------------------------
        A real gate has blue ONLY at the two posts. A blue tarp, banner, or
        barrier has blue that extends continuously well beyond the posts in
        the horizontal direction. By sampling the mask in a window just
        outside each post's outer edge, we directly measure whether the
        blue region is confined (real gate) or extending (false positive).

        Measurement
        -----------
        Sample at 3 heights within the post overlap zone. At each height,
        read a window of width = 25% of the opening width outside each post.
        The isolation score is 1 - max(left_extension, right_extension).

        Expected values
        ---------------
        Real gate:    left/right extension ≈ 0.0-0.1  → isolation ≈ 0.90-1.00
        Blue banner:  left/right extension ≈ 0.7-1.0  → isolation ≈ 0.00-0.30
        """
        cfg = self.config
        opening_w = max(1, right["x"] - (left["x"] + left["w"]))
        check_w = max(12, int(cfg.isolation_check_width_ratio * opening_w))

        y_top_overlap = max(left["y"], right["y"])
        y_bot_overlap = min(left["y"] + left["h"], right["y"] + right["h"])
        if y_bot_overlap <= y_top_overlap + 4:
            return 0.5   # no overlap — cannot assess

        sample_ys = [
            int(y_top_overlap + 0.2 * (y_bot_overlap - y_top_overlap)),
            int(y_top_overlap + 0.5 * (y_bot_overlap - y_top_overlap)),
            int(y_top_overlap + 0.8 * (y_bot_overlap - y_top_overlap)),
        ]

        isolation_vals: list[float] = []
        for y in sample_ys:
            if not (0 <= y < image_h):
                continue

            # Left extension: sample to the left of the left post
            lx1 = max(0, left["x"] - check_w)
            lx2 = left["x"]
            left_ext = float(mask[y, lx1:lx2].mean() / 255.0) if lx2 > lx1 else 0.0

            # Right extension: sample to the right of the right post
            rx1 = right["x"] + right["w"]
            rx2 = min(image_w, rx1 + check_w)
            right_ext = float(mask[y, rx1:rx2].mean() / 255.0) if rx2 > rx1 else 0.0

            isolation_vals.append(1.0 - max(left_ext, right_ext))

        if not isolation_vals:
            return 0.5
        return float(np.clip(np.mean(isolation_vals), 0.0, 1.0))

    def _score_checkered_top(
        self,
        gray: np.ndarray,
        x1: int, x2: int,
        y_top: int,
        pair_h: int,
    ) -> float:
        """Detect the competition gate's checkered top bar.

        The alternating black-white pattern produces sparse but sharp column
        transitions in the mean profile.  We use the 90th-percentile of
        |diff(col_means)| rather than the mean so that a handful of large
        transitions (real checker) are not drowned out by hundreds of
        flat regions between transition edges.

        Real gate:      p90 ≈ 50-150  →  score ≈ 0.35-1.0
        Random objects: p90 ≈ 2-15    →  score ≈ 0.01-0.10
        """
        img_h = gray.shape[0]
        band_h = max(8, int(0.07 * pair_h))
        y0 = max(0, y_top - band_h)
        y1_ = min(img_h, y_top + band_h // 3)
        x1c = max(0, x1)
        x2c = min(gray.shape[1], x2)

        strip = gray[y0:y1_, x1c:x2c]
        if strip.size < 16:
            return 0.0

        col_means = strip.mean(axis=0).astype(np.float32)
        if len(col_means) < 4:
            return 0.0

        diffs = np.diff(col_means)
        if len(diffs) < 2:
            return 0.0

        abs_diffs = np.abs(diffs)
        # 90th percentile: captures the actual transition magnitude,
        # not diluted by the flat within-square regions
        p90 = float(np.percentile(abs_diffs, 90))
        sign_changes = float(np.sum(np.abs(np.diff(np.sign(diffs))) > 0))
        alternation = 0.5 + 0.5 * sign_changes / max(len(diffs) - 1, 1)

        # Normalize: p90 ≈ 120 for a sharp black/white checker at full contrast
        return float(np.clip(p90 / 120.0 * alternation, 0.0, 1.0))

    def _score_corner_responses(
        self, gray: np.ndarray, corners: np.ndarray
    ) -> float:
        """Shi-Tomasi corner-response score at the four predicted opening corners.

        Real gate corners (intersection of post edge and top bar) produce
        strong local eigenvalues in the structure tensor, while the corners
        of a false-positive bounding box typically do not.
        """
        corner_map = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
        h_img, w_img = gray.shape
        scores: list[float] = []
        for pt in corners.reshape(4, 2):
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            x0, x1 = max(0, x - 6), min(w_img, x + 7)
            y0, y1 = max(0, y - 6), min(h_img, y + 7)
            patch = corner_map[y0:y1, x0:x1]
            if patch.size > 0:
                scores.append(float(patch.max()))
        if not scores:
            return 0.0
        # Strong gate corners yield responses of ~400-2000; normalise to [0, 1]
        return float(np.clip(float(np.mean(scores)) / 700.0, 0.0, 1.0))

    def _score_gradient_perpendicularity(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        post: dict[str, Any],
        side: str,
    ) -> float:
        """Gradient perpendicularity at the post's inner (opening-facing) edge.

        A vertical blue stripe has strongly *horizontal* image gradients at
        its left and right boundaries.  Measuring the cosine alignment between
        the Sobel-X gradient and the expected inward direction gives a cheap,
        physics-grounded confirmation that a candidate blob is really a post.

        side='left'  → inner edge is right side of left post  → expect gx < 0
        side='right' → inner edge is left side of right post  → expect gx > 0
        """
        h_img, w_img = gx.shape
        if side == "left":
            ex = int(np.clip(post["x"] + post["w"], 0, w_img - 1))
            expected = -1.0
        else:
            ex = int(np.clip(post["x"], 0, w_img - 1))
            expected = +1.0

        y_top = max(0, post["y"])
        y_bot = min(h_img, post["y"] + post["h"])
        if y_bot <= y_top + 2:
            return 0.0

        n = max(3, min(12, (y_bot - y_top) // 3))
        y_samples = np.linspace(y_top + 2, y_bot - 2, n).astype(int)

        perp_vals: list[float] = []
        for y in y_samples:
            if 0 <= y < h_img:
                gx_v = float(gx[y, ex])
                gy_v = float(gy[y, ex])
                mag  = float(np.hypot(gx_v, gy_v))
                if mag < 8.0:
                    continue
                perp_vals.append(max(0.0, expected * gx_v / mag))

        return float(np.mean(perp_vals)) if perp_vals else 0.0

    def _score_horizontal_profile(
        self,
        mask: np.ndarray,
        left: dict[str, Any],
        right: dict[str, Any],
        image_w: int,
    ) -> float:
        """Horizontal scanline profile: post | dark gap | post.

        This is the single most discriminative feature distinguishing a real
        gate (two separated posts flanking an empty opening) from false
        positives like blue walls or partial floor markings.  At the vertical
        midpoint of the overlap region, the expected mask profile is:
            low | high | low | high | low
        We measure the contrast between post and opening regions directly.
        """
        mid_y = int(0.5 * (
            max(left["y"], right["y"])
            + min(left["y"] + left["h"], right["y"] + right["h"])
        ))
        x_start = max(0, left["x"] - 4)
        x_end   = min(image_w, right["x"] + right["w"] + 4)

        if mid_y < 0 or mid_y >= mask.shape[0] or x_end <= x_start:
            return 0.0

        scanline = mask[mid_y, x_start:x_end].astype(np.float32) / 255.0
        if len(scanline) < 8:
            return 0.0

        lpost_end   = int(np.clip(left["x"]  + left["w"]  - x_start, 0, len(scanline)))
        rpost_start = int(np.clip(right["x"]              - x_start, 0, len(scanline)))

        if rpost_start <= lpost_end:
            return 0.0

        post_region    = np.concatenate([scanline[:lpost_end], scanline[rpost_start:]])
        opening_region = scanline[lpost_end:rpost_start]

        if len(post_region) == 0 or len(opening_region) == 0:
            return 0.0

        contrast = float(post_region.mean()) - float(opening_region.mean())
        return float(np.clip(contrast / 0.45, 0.0, 1.0))

    @staticmethod
    def _perspective_aspect_score(
        opening_w: float,
        opening_h: float,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> float:
        """Opening-aspect score corrected for viewing-angle perspective.

        A square gate (1.4 × 1.4 m) appears square (aspect ≈ 1.0) straight-on
        but compressed horizontally when viewed at yaw.  The existing fixed
        target of 0.68 penalises any gate seen from more than ~15° off-axis.
        We estimate yaw from the asymmetry of post widths and heights and shift
        the target accordingly.
        """
        opening_aspect = opening_w / max(opening_h, 1.0)

        lw = max(left.get("w", 1), 1)
        rw = max(right.get("w", 1), 1)
        lh = max(left.get("h", 1), 1)
        rh = max(right.get("h", 1), 1)
        width_asym  = abs(lw - rw) / float(lw + rw)
        height_asym = abs(lh - rh) / float(lh + rh)
        yaw_est = min(0.65 * width_asym + 0.35 * height_asym, 0.60)

        # For a square gate: projected opening width ∝ cos(yaw).
        # Corrected target = nominal / (1 - yaw_compression)
        corrected_target = 0.68 / max(1.0 - 0.50 * yaw_est, 0.50)
        radius = 0.80 + 0.40 * yaw_est  # wider tolerance at high yaw

        return float(max(0.0, 1.0 - abs(opening_aspect - corrected_target) / radius))

    # ── Detection internals ─────────────────────────────────────────────────

    def _select_candidate(
        self,
        ring_candidate: Optional[GateDetection],
        post_candidate: Optional[GateDetection],
    ) -> Optional[GateDetection]:
        if post_candidate is not None and post_candidate.is_valid:
            if (
                ring_candidate is not None
                and ring_candidate.is_valid
                and ring_candidate.confidence >= max(
                    self.config.strong_ring_confidence,
                    post_candidate.confidence + self.config.ring_selection_margin,
                )
            ):
                return ring_candidate
            return post_candidate

        if (
            ring_candidate is not None
            and ring_candidate.is_valid
            and ring_candidate.confidence >= self.config.strong_ring_confidence
        ):
            return ring_candidate

        candidates = [c for c in (post_candidate, ring_candidate) if c is not None]
        if not candidates:
            return None

        if post_candidate is not None and ring_candidate is not None:
            if post_candidate.confidence >= ring_candidate.confidence - self.config.ring_selection_margin:
                return post_candidate

        return max(candidates, key=lambda d: d.confidence)

    def _merge_nearby_verticals(
        self, comps: list[dict[str, Any]], image_w: int
    ) -> list[dict[str, Any]]:
        if not comps:
            return comps
        comps = sorted(comps, key=lambda c: c["cx"] if "cx" in c else c["x"])
        merged: list[dict[str, Any]] = []
        max_x_gap = max(6, int(0.025 * image_w))

        for comp in comps:
            cur = dict(comp)
            if not merged:
                merged.append(cur)
                continue
            prev = merged[-1]
            x_gap = cur["x"] - (prev["x"] + prev["w"])
            y_overlap = max(
                0,
                min(prev["y"] + prev["h"], cur["y"] + cur["h"]) - max(prev["y"], cur["y"]),
            )
            overlap_ratio = y_overlap / max(min(prev["h"], cur["h"]), 1)
            if x_gap <= max_x_gap and overlap_ratio >= 0.25:
                x1 = min(prev["x"], cur["x"])
                y1 = min(prev["y"], cur["y"])
                x2 = max(prev["x"] + prev["w"], cur["x"] + cur["w"])
                y2 = max(prev["y"] + prev["h"], cur["y"] + cur["h"])
                bw = x2 - x1
                bh = y2 - y1
                merged[-1] = {
                    "x":     int(x1),
                    "y":     int(y1),
                    "w":     int(bw),
                    "h":     int(bh),
                    "area":  int(prev["area"] + cur["area"]),
                    "cx":    float(x1 + 0.5 * bw),
                    "cy":    float(y1 + 0.5 * bh),
                    "aspect": float(bh / max(bw, 1)),
                    "fill":  float(0.5 * (prev.get("fill", 0.5) + cur.get("fill", 0.5))),
                    "source": "merged",
                }
            else:
                merged.append(cur)
        return merged

    def _detect_post_pair_hypothesis(
        self,
        mask: np.ndarray,
        edges: np.ndarray,
        gray: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        frame_hsv: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> Optional[GateDetection]:
        # ── Candidate vertical blobs ───────────────────────────────────────
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        verticals: list[dict[str, Any]] = []

        for i in range(1, n_labels):
            x, y, bw, bh, area = stats[i]
            if area < self.config.min_component_area:
                continue
            aspect = bh / max(bw, 1)
            if aspect < self.config.min_vertical_aspect:
                continue
            if aspect > self.config.max_post_aspect:
                continue  # too thin — person's leg, thin rod, not a gate post
            if bh < self.config.min_vertical_height_ratio * image_h:
                continue
            fill = float(mask[y:y + bh, x:x + bw].mean() / 255.0) if bw > 0 and bh > 0 else 0.0
            if fill < 0.16:
                continue
            verticals.append({
                "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                "area": int(area),
                "cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
                "aspect": float(aspect), "fill": fill,
                "source": "cc",
            })

        # Supplement with Hough-detected verticals (fragmented poles)
        if self.config.enable_hough_verticals:
            for hp in self._hough_vertical_segments(mask, image_h, image_w):
                hp_aspect = hp["h"] / max(hp["w"], 1)
                if hp_aspect > self.config.max_post_aspect:
                    continue   # still too thin even as a Hough cluster
                covered = any(
                    abs(hp["cx"] - v["cx"]) < 20
                    and abs(hp["cy"] - v["cy"]) < v["h"] * 0.45
                    for v in verticals
                )
                if not covered:
                    verticals.append(hp)

        verticals = self._merge_nearby_verticals(verticals, image_w)
        verticals.sort(key=lambda c: c["cx"])

        best_det: Optional[GateDetection] = None
        best_score: float = 0.0

        min_opening_w_px = max(20, int(self.config.min_opening_width_ratio  * image_w))
        min_opening_h_px = max(22, int(self.config.min_opening_height_ratio * image_h))
        cfg = self.config

        for i in range(len(verticals)):
            for j in range(i + 1, len(verticals)):
                left  = verticals[i]
                right = verticals[j]

                gap = right["x"] - (left["x"] + left["w"])
                if gap < 12 or gap > 0.82 * image_w:
                    continue

                # Minimum gap proportional to post width (single-pole-split suppression).
                # A pole that the segmentation splits into two halves has gap ≈ 0.
                # A real gate pair has gap >> post width.
                max_post_w = max(left["w"], right["w"], 1)
                if gap < cfg.min_gap_to_post_width * max_post_w:
                    continue

                # Reject if either blob is too thin to be a post (person legs,
                # thin rods, Hough artefacts). Checked here as a second fence
                # after the CC/Hough per-blob filters.
                if left["h"]  / max(left["w"],  1) > cfg.max_post_aspect:
                    continue
                if right["h"] / max(right["w"], 1) > cfg.max_post_aspect:
                    continue

                # Vertical overlap
                vert_overlap = max(
                    0,
                    min(left["y"] + left["h"], right["y"] + right["h"]) - max(left["y"], right["y"]),
                )
                overlap_ratio = vert_overlap / max(min(left["h"], right["h"]), 1)
                height_ratio  = min(left["h"], right["h"]) / max(left["h"], right["h"])
                width_ratio   = min(left["w"], right["w"]) / max(left["w"], right["w"])

                # ── Geometry ───────────────────────────────────────────────
                top_y    = int(min(left["y"], right["y"]))
                bottom_y = int(max(left["y"] + left["h"], right["y"] + right["h"]))
                x_outer1 = int(left["x"])
                x_outer2 = int(right["x"] + right["w"])
                x_inner1 = int(left["x"] + left["w"])
                x_inner2 = int(right["x"])

                if x_inner2 <= x_inner1:
                    continue

                pair_h    = max(1, bottom_y - top_y)
                opening_w = x_inner2 - x_inner1
                opening_h = pair_h

                if opening_w < min_opening_w_px or opening_h < min_opening_h_px:
                    continue

                # ── Hard aspect ratio cutoffs (banner / tarp suppression) ──
                # This is the primary fix for detecting blue barriers when
                # no gate is present.  Real gate openings: aspect 0.6-1.8.
                # Blue banner/tarp "post pairs": aspect 2.5-14.
                opening_aspect_raw = opening_w / max(opening_h, 1.0)
                if opening_aspect_raw > cfg.max_opening_aspect:
                    continue
                outer_aspect_raw = (x_outer2 - x_outer1) / max(pair_h, 1.0)
                if outer_aspect_raw > cfg.max_outer_aspect:
                    continue

                # ── Soft geometric penalties ───────────────────────────────
                overlap_penalty  = max(0.0, cfg.soft_overlap_floor      - overlap_ratio) * 0.45
                height_penalty   = max(0.0, cfg.soft_height_ratio_floor - height_ratio)  * 0.35

                # Opening hollowness
                mx = max(2, int(0.12 * opening_w))
                my = max(2, int(0.08 * opening_h))
                cx1 = min(image_w, x_inner1 + mx)
                cx2 = max(cx1 + 1, min(image_w, x_inner2 - mx))
                cy1 = min(image_h, top_y + my)
                cy2 = max(cy1 + 1, min(image_h, bottom_y - my))
                core = mask[cy1:cy2, cx1:cx2]
                opening_occupancy = float(core.mean() / 255.0) if core.size else 1.0
                occupancy_penalty = max(0.0, opening_occupancy - cfg.soft_occupancy_ceiling) * 0.50

                if opening_occupancy > cfg.hard_occupancy_ceiling:
                    continue

                hollow_score = max(0.0, 1.0 - opening_occupancy / 0.22)
                if hollow_score < cfg.min_hollow_score:
                    continue

                # ── Scoring terms ──────────────────────────────────────────

                # 1. Symmetry
                symmetry_score = 0.60 * height_ratio + 0.40 * width_ratio

                # 2. Perspective-corrected aspect
                opening_aspect_score = self._perspective_aspect_score(
                    float(opening_w), float(opening_h), left, right
                )

                # 3. Gap
                gap_ratio  = gap / max(0.5 * (left["h"] + right["h"]), 1.0)
                gap_score  = max(0.0, 1.0 - abs(gap_ratio - 0.70) / 1.20)

                # 4. Top support: blend checkered-bar score with Canny density
                checker_score = self._score_checkered_top(gray, x_outer1, x_outer2, top_y, pair_h)

                # Hard gate: the checkered top bar is the most gate-specific feature
                # in the entire competition environment. checker=0.01 means zero
                # alternating pattern above the posts — not a gate.
                if checker_score < cfg.min_checker_score:
                    continue

                band          = max(4, int(0.05 * pair_h))
                top_band      = edges[
                    max(0, top_y - band): min(image_h, top_y + band),
                    x_outer1: x_outer2,
                ]
                edge_density      = float(top_band.mean() / 255.0) if top_band.size else 0.0
                top_support_score = 0.55 * checker_score + 0.45 * min(1.0, edge_density / 0.14)

                if top_support_score < cfg.min_top_support_score:
                    continue

                # 5. Post fill
                lp = mask[left["y"]: left["y"] + left["h"], left["x"]: left["x"] + left["w"]]
                rp = mask[right["y"]:right["y"] + right["h"], right["x"]:right["x"] + right["w"]]
                left_fill  = float(lp.mean() / 255.0) if lp.size else 0.0
                right_fill = float(rp.mean() / 255.0) if rp.size else 0.0
                post_fill_score = min(1.0, 0.5 * (left_fill + right_fill) / 0.55)

                # 6. Balance / centre
                balance_score = max(
                    0.0,
                    1.0 - abs(left_fill - right_fill) / max(left_fill + right_fill, 1e-6),
                )
                center_x     = x_inner1 + 0.5 * opening_w
                center_score = max(0.0, 1.0 - abs(center_x - 0.5 * image_w) / (0.55 * image_w))

                # 7. NEW — Shi-Tomasi corner responses
                pred_corners = np.array([
                    [x_inner1, top_y],
                    [x_inner2, top_y],
                    [x_inner2, bottom_y],
                    [x_inner1, bottom_y],
                ], dtype=np.float32)
                corner_response_score = self._score_corner_responses(gray, pred_corners)

                # 8. NEW — Gradient perpendicularity at post inner edges
                perp_l = self._score_gradient_perpendicularity(gx, gy, left,  side="left")
                perp_r = self._score_gradient_perpendicularity(gx, gy, right, side="right")
                gradient_perp_score = 0.5 * (perp_l + perp_r)

                # 9. NEW — Horizontal scanline profile
                profile_score = self._score_horizontal_profile(mask, left, right, image_w)

                # 10. NEW — Raw color purity (orange/cream veto)
                # Computed from raw HSV, independent of mask or model state.
                left_purity  = self._score_post_color_purity(frame_hsv, left)
                right_purity = self._score_post_color_purity(frame_hsv, right)
                color_purity_score = 0.5 * (left_purity + right_purity)

                # Hard veto: if either post has essentially no blue pixels,
                # reject the pair immediately regardless of geometry score.
                if left_purity  < cfg.post_color_purity_hard_floor:
                    continue
                if right_purity < cfg.post_color_purity_hard_floor:
                    continue

                # Soft penalty for low-purity posts (partially shadowed, etc.)
                purity_penalty = max(0.0, cfg.post_color_purity_soft_floor - color_purity_score) * 0.50

                # 11. NEW — Blue isolation (banner suppression)
                # Real posts are isolated blue blobs; banners are continuous.
                isolation_score = self._score_blue_isolation(mask, left, right, image_w, image_h)

                # Hard veto: if blue extends strongly past both posts → banner
                if isolation_score < cfg.isolation_hard_floor:
                    continue

                # Soft penalty for partial extension
                isolation_penalty = max(0.0, cfg.isolation_soft_floor - isolation_score) * 0.40

                # Edge penalty
                edge_penalty = 0.0
                if left["x"] < 4 or right["x"] + right["w"] > image_w - 4:
                    edge_penalty += 0.06
                if (x_outer2 - x_outer1) < 0.12 * image_w:
                    edge_penalty += 0.04

                # ── Combined score ─────────────────────────────────────────
                score = (
                    cfg.w_symmetry        * symmetry_score
                    + cfg.w_overlap       * overlap_ratio
                    + cfg.w_gap           * gap_score
                    + cfg.w_aspect        * opening_aspect_score
                    + cfg.w_top_support   * top_support_score
                    + cfg.w_hollow        * hollow_score
                    + cfg.w_corner_response * corner_response_score
                    + cfg.w_gradient_perp * gradient_perp_score
                    + cfg.w_profile       * profile_score
                    + cfg.w_color_purity  * color_purity_score
                    + cfg.w_isolation     * isolation_score
                    + cfg.w_post_fill     * post_fill_score
                    + cfg.w_balance       * balance_score
                    + cfg.w_center        * center_score
                    - edge_penalty
                    - overlap_penalty
                    - height_penalty
                    - occupancy_penalty
                    - purity_penalty
                    - isolation_penalty
                )

                if score <= best_score:
                    continue

                outer_bbox   = (x_outer1, top_y, x_outer2 - x_outer1, bottom_y - top_y)
                opening_bbox = (x_inner1, top_y, x_inner2 - x_inner1, bottom_y - top_y)
                cx_open = float(x_inner1 + 0.5 * opening_w)
                cy_open = float(top_y + 0.5 * opening_h)

                best_score = float(score)
                best_det = GateDetection(
                    is_valid=score >= cfg.min_gate_confidence,
                    confidence=float(score),
                    mode="post_pair",
                    image_size=(image_w, image_h),
                    center_px=(cx_open, cy_open),
                    outer_bbox=outer_bbox,
                    opening_bbox=opening_bbox,
                    corners_px=pred_corners,
                    opening_width_px=float(opening_w),
                    opening_height_px=float(opening_h),
                    debug={
                        "left": left,
                        "right": right,
                        "height_ratio": height_ratio,
                        "width_ratio": width_ratio,
                        "overlap_ratio": overlap_ratio,
                        "gap_ratio": gap_ratio,
                        "opening_aspect": float(opening_w / max(opening_h, 1)),
                        "outer_aspect": float(outer_aspect_raw),
                        "opening_aspect_score": opening_aspect_score,
                        "top_support_score": top_support_score,
                        "checker_score": checker_score,
                        "hollow_score": hollow_score,
                        "corner_response_score": corner_response_score,
                        "gradient_perp_score": gradient_perp_score,
                        "profile_score": profile_score,
                        "color_purity_score": color_purity_score,
                        "left_purity": left_purity,
                        "right_purity": right_purity,
                        "isolation_score": isolation_score,
                        "opening_occupancy": opening_occupancy,
                        "post_fill_score": post_fill_score,
                        "soft_penalties": {
                            "overlap": overlap_penalty,
                            "height": height_penalty,
                            "occupancy": occupancy_penalty,
                            "purity": purity_penalty,
                            "isolation": isolation_penalty,
                        },
                    },
                )

        return best_det

    def _detect_ring_hypothesis(
        self, mask: np.ndarray, image_w: int, image_h: int
    ) -> Optional[GateDetection]:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None
        hierarchy = hierarchy[0]
        best_det: Optional[GateDetection] = None
        best_score = 0.0

        for idx, contour in enumerate(contours):
            outer_area = cv2.contourArea(contour)
            if outer_area < 1400:
                continue
            child = hierarchy[idx][2]
            if child == -1:
                continue
            hole_contours = []
            j = child
            while j != -1:
                if cv2.contourArea(contours[j]) > 650:
                    hole_contours.append(contours[j])
                j = hierarchy[j][0]
            if not hole_contours:
                continue

            ox, oy, ow, oh = cv2.boundingRect(contour)
            if ow / max(oh, 1) < 0.65 or ow / max(oh, 1) > 1.60:
                continue

            inner_pts = np.vstack(hole_contours)
            ix, iy, iw, ih = cv2.boundingRect(inner_pts)
            if iw < max(16, int(self.config.min_opening_width_ratio  * image_w)):
                continue
            if ih < max(20, int(self.config.min_opening_height_ratio * image_h)):
                continue

            hole_area = float(sum(cv2.contourArea(h) for h in hole_contours))
            hole_ratio = hole_area / max(outer_area, 1.0)
            opening_aspect = iw / max(ih, 1)

            centeredness = 1.0 - (
                abs((ix + 0.5 * iw) - (ox + 0.5 * ow)) / max(ow, 1)
                + abs((iy + 0.5 * ih) - (oy + 0.5 * oh)) / max(oh, 1)
            )
            centeredness = max(0.0, centeredness)
            if centeredness < 0.22:
                continue

            aspect_score     = max(0.0, 1.0 - abs(opening_aspect - 0.95) / 0.65)
            hole_ratio_score = max(0.0, 1.0 - abs(hole_ratio - 0.30) / 0.28)
            size_score       = min(1.0, (ow * oh) / max(0.12 * image_w * image_h, 1.0))
            score = (
                0.33 * aspect_score
                + 0.29 * hole_ratio_score
                + 0.23 * centeredness
                + 0.15 * size_score
            )
            if score <= best_score:
                continue

            rect    = cv2.minAreaRect(inner_pts)
            corners = cv2.boxPoints(rect).astype(np.float32)
            best_score = float(score)
            best_det = GateDetection(
                is_valid=score >= self.config.min_gate_confidence,
                confidence=float(score),
                mode="ring",
                image_size=(image_w, image_h),
                center_px=(ix + 0.5 * iw, iy + 0.5 * ih),
                outer_bbox=(ox, oy, ow, oh),
                opening_bbox=(ix, iy, iw, ih),
                corners_px=corners,
                opening_width_px=float(iw),
                opening_height_px=float(ih),
                debug={
                    "outer_area": float(outer_area),
                    "hole_area": hole_area,
                    "hole_ratio": hole_ratio,
                    "opening_aspect": opening_aspect,
                    "centeredness": centeredness,
                },
            )
        return best_det

    def _add_control_quantities(self, det: GateDetection) -> GateDetection:
        image_w, image_h = det.image_size

        if det.center_px is not None:
            cx, cy = det.center_px
            det.lateral_error_norm = float((cx - 0.5 * image_w) / max(0.5 * image_w, 1.0))
            det.vertical_error_norm = float((cy - 0.5 * image_h) / max(0.5 * image_h, 1.0))

        if det.opening_bbox is not None:
            _, _, bw, bh = det.opening_bbox
            det.opening_width_px  = float(bw)
            det.opening_height_px = float(bh)

        if det.mode == "post_pair" and "left" in det.debug and "right" in det.debug:
            left  = det.debug["left"]
            right = det.debug["right"]
            wa = (right["w"] - left["w"]) / max(right["w"] + left["w"], 1)
            ha = (right["h"] - left["h"]) / max(right["h"] + left["h"], 1)
            det.yaw_proxy = float(0.65 * wa + 0.35 * ha)
        else:
            det.yaw_proxy = 0.0

        if (
            self.config.gate_opening_width_m is not None
            and self.config.fx_px is not None
            and det.opening_width_px is not None
            and det.opening_width_px > 1.0
        ):
            det.range_estimate_m = float(
                self.config.fx_px * self.config.gate_opening_width_m / det.opening_width_px
            )
        return det

    def _scale_detection(self, det: GateDetection, scale: float) -> GateDetection:
        """Return a copy of `det` with all pixel quantities multiplied by `scale`."""
        def _sb(b: Optional[tuple]) -> Optional[tuple]:
            if b is None:
                return None
            x, y, w, h = b
            return (
                int(round(x * scale)), int(round(y * scale)),
                int(round(w * scale)), int(round(h * scale)),
            )

        iw, ih = det.image_size
        center = (
            (det.center_px[0] * scale, det.center_px[1] * scale)
            if det.center_px is not None else None
        )

        # Deep-copy debug and rescale post sub-dicts
        dbg = dict(det.debug)
        for key in ("left", "right"):
            if key in dbg and isinstance(dbg[key], dict):
                d = dict(dbg[key])
                for k in ("x", "y", "w", "h"):
                    if k in d:
                        d[k] = int(round(d[k] * scale))
                for k in ("cx", "cy"):
                    if k in d:
                        d[k] = float(d[k]) * scale
                dbg[key] = d

        return GateDetection(
            is_valid=det.is_valid,
            confidence=det.confidence,
            mode=det.mode,
            image_size=(int(iw * scale), int(ih * scale)),
            center_px=center,
            outer_bbox=_sb(det.outer_bbox),
            opening_bbox=_sb(det.opening_bbox),
            corners_px=det.corners_px * scale if det.corners_px is not None else None,
            opening_width_px=(det.opening_width_px  * scale) if det.opening_width_px  is not None else None,
            opening_height_px=(det.opening_height_px * scale) if det.opening_height_px is not None else None,
            debug=dbg,
        )

    def _can_estimate_pose(self, det: GateDetection) -> bool:
        return (
            det.corners_px is not None
            and self.config.gate_opening_width_m  is not None
            and self.config.gate_opening_height_m is not None
            and self.config.camera_matrix is not None
            and self.config.dist_coeffs   is not None
        )

    def _estimate_pose(self, det: GateDetection) -> None:
        hw = 0.5 * float(self.config.gate_opening_width_m)   # type: ignore[arg-type]
        hh = 0.5 * float(self.config.gate_opening_height_m)  # type: ignore[arg-type]
        object_points = np.array([
            [-hw, -hh, 0.0],
            [ hw, -hh, 0.0],
            [ hw,  hh, 0.0],
            [-hw,  hh, 0.0],
        ], dtype=np.float32)
        image_points = self._order_corners(det.corners_px.astype(np.float32))  # type: ignore[union-attr]
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            self.config.camera_matrix, self.config.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if ok:
            det.pose_rvec = rvec
            det.pose_tvec = tvec
            det.range_estimate_m = float(np.linalg.norm(tvec))

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        pts = corners.reshape(4, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(d)]
        ordered[3] = pts[np.argmax(d)]
        return ordered


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_image_paths(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run enhanced gate detector on an image folder.")
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("--save-dir",         type=Path,  default=None)
    parser.add_argument("--min-confidence",   type=float, default=0.50)
    parser.add_argument("--gate-width-m",     type=float, default=None)
    parser.add_argument("--fx-px",            type=float, default=None)
    parser.add_argument("--no-temporal",      action="store_true")
    parser.add_argument("--no-hough",         action="store_true")
    parser.add_argument("--no-multiscale",    action="store_true")
    parser.add_argument("--no-adaptive-color",action="store_true")
    args = parser.parse_args()

    cfg = GateDetectorConfig(
        min_gate_confidence=args.min_confidence,
        gate_opening_width_m=args.gate_width_m,
        fx_px=args.fx_px,
        enable_temporal_mask=not args.no_temporal,
        enable_hough_verticals=not args.no_hough,
        enable_multiscale=not args.no_multiscale,
        enable_adaptive_color_model=not args.no_adaptive_color,
    )
    detector = EnhancedGateDetector(cfg)

    paths = load_image_paths(args.image_folder)
    if not paths:
        raise SystemExit(f"No images found in {args.image_folder}")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        proc, mask, det = detector.detect(frame)
        valid_count += int(det.is_valid)

        if args.save_dir is not None:
            annotated = detector.annotate(proc, det, mask)
            cv2.imwrite(str(args.save_dir / path.name), annotated)

        print(
            f"{path.name}\t"
            f"valid={det.is_valid}\t"
            f"conf={det.confidence:.3f}\t"
            f"mode={det.mode}\t"
            f"center={det.center_px}\t"
            f"cm_updates={detector._color_model.update_count}"
        )

    print(f"\nDetected {valid_count}/{len(paths)} valid frames  "
          f"({100*valid_count/max(len(paths),1):.1f}%)")
