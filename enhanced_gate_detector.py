from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import math

import cv2
import numpy as np


@dataclass
class GateDetectorConfig:
    rotate_portrait_frames: bool = True
    min_component_area: int = 180
    min_vertical_aspect: float = 1.2
    min_vertical_height_ratio: float = 0.08
    min_gate_confidence: float = 0.50
    strong_ring_confidence: float = 0.72
    hsv_lower_blue: tuple[int, int, int] = (92, 48, 22)
    hsv_upper_blue: tuple[int, int, int] = (138, 255, 255)
    min_bg_diff: int = 24
    min_br_diff: int = 18
    min_saturation_for_rgb_mask: int = 42
    gate_opening_width_m: Optional[float] = None
    gate_opening_height_m: Optional[float] = None
    fx_px: Optional[float] = None
    fy_px: Optional[float] = None
    cx_px: Optional[float] = None
    cy_px: Optional[float] = None
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None


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


class AlphaBetaGateTracker:
    """Lightweight constant-velocity tracker.

    This is intentionally simple: for flight control, a smooth and conservative
    estimate is more valuable than a fragile overfit tracker.
    """

    def __init__(self, alpha: float = 0.70, beta: float = 0.25, confidence_decay: float = 0.88):
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

    def update(self, detection: GateDetection, dt: float = 1 / 30.0) -> TrackerState:
        if not self.state.valid:
            if detection.is_valid and detection.opening_bbox is not None:
                x, y, w, h = detection.opening_bbox
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

        if not detection.is_valid or detection.opening_bbox is None:
            return self.state

        x, y, w, h = detection.opening_bbox
        zx = x + 0.5 * w
        zy = y + 0.5 * h
        zw = float(w)
        zh = float(h)

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

        self.state.confidence = max(self.state.confidence, float(detection.confidence))
        return self.state


class EnhancedGateDetector:
    """Robust gate detector aimed at flight-control integration.

    Design philosophy:
    - Conservative over optimistic: false positives are more dangerous than misses.
    - Geometry-aware: the output is not only a bounding box but also center,
      opening size, normalized errors, and a yaw proxy.
    - Control-ready: returns quantities directly usable by a guidance controller.
    - Extensible: can use solvePnP when the camera is calibrated.
    """

    def __init__(self, config: Optional[GateDetectorConfig] = None):
        self.config = config or GateDetectorConfig()

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = frame_bgr.copy()
        if self.config.rotate_portrait_frames and frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame

    def segment_gate_blue(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(frame_bgr)
        b = b.astype(np.int16)
        g = g.astype(np.int16)
        r = r.astype(np.int16)

        mask_hsv = cv2.inRange(
            hsv,
            np.array(self.config.hsv_lower_blue, dtype=np.uint8),
            np.array(self.config.hsv_upper_blue, dtype=np.uint8),
        )
        mask_rgb = (
            (b - g > self.config.min_bg_diff)
            & (b - r > self.config.min_br_diff)
            & (hsv[:, :, 1] > self.config.min_saturation_for_rgb_mask)
        ).astype(np.uint8) * 255

        mask = cv2.bitwise_and(mask_hsv, mask_rgb)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, GateDetection]:
        frame = self.preprocess(frame_bgr)
        mask = self.segment_gate_blue(frame)
        H, W = frame.shape[:2]

        ring_candidate = self._detect_ring_hypothesis(mask, W, H)
        post_candidate = self._detect_post_pair_hypothesis(mask, W, H)

        selected = None
        if ring_candidate is not None and ring_candidate.confidence >= self.config.strong_ring_confidence:
            selected = ring_candidate
        elif post_candidate is not None:
            selected = post_candidate
        elif ring_candidate is not None:
            selected = ring_candidate

        if selected is None:
            return frame, mask, GateDetection(
                is_valid=False,
                confidence=0.0,
                mode="none",
                image_size=(W, H),
                debug={"reason": "no geometric hypothesis survived gating"},
            )

        selected = self._add_control_quantities(selected)
        if self._can_estimate_pose(selected):
            self._estimate_pose(selected)
        return frame, mask, selected

    def annotate(self, frame_bgr: np.ndarray, detection: GateDetection, mask: Optional[np.ndarray] = None) -> np.ndarray:
        vis = frame_bgr.copy()
        H, W = vis.shape[:2]

        if mask is not None:
            mask_small = cv2.resize(mask, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
            mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            vis[0 : mask_small.shape[0], 0 : mask_small.shape[1]] = mask_small

        if detection.outer_bbox is not None:
            x, y, w, h = detection.outer_bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)

        if detection.opening_bbox is not None:
            x, y, w, h = detection.opening_bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if detection.corners_px is not None:
            corners = detection.corners_px.astype(int).reshape(-1, 1, 2)
            cv2.polylines(vis, [corners], True, (255, 255, 0), 2)

        if detection.center_px is not None:
            cx, cy = map(int, detection.center_px)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(vis, (W // 2, H // 2), (cx, cy), (0, 0, 255), 1)

        status = f"{detection.mode} conf={detection.confidence:.2f}"
        if detection.lateral_error_norm is not None and detection.vertical_error_norm is not None:
            status += f" ex={detection.lateral_error_norm:+.2f} ey={detection.vertical_error_norm:+.2f}"
        if detection.range_estimate_m is not None:
            status += f" range~{detection.range_estimate_m:.2f}m"

        cv2.putText(vis, status, (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return vis

    def _detect_ring_hypothesis(self, mask: np.ndarray, W: int, H: int) -> Optional[GateDetection]:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None

        hierarchy = hierarchy[0]
        best_det = None
        best_score = 0.0

        for idx, contour in enumerate(contours):
            outer_area = cv2.contourArea(contour)
            if outer_area < 1600:
                continue

            child = hierarchy[idx][2]
            if child == -1:
                continue

            hole_contours = []
            j = child
            while j != -1:
                area = cv2.contourArea(contours[j])
                if area > 700:
                    hole_contours.append(contours[j])
                j = hierarchy[j][0]

            if not hole_contours:
                continue

            outer_bbox = cv2.boundingRect(contour)
            outer_x, outer_y, outer_w, outer_h = outer_bbox
            outer_aspect = outer_w / max(outer_h, 1)
            if outer_aspect < 0.55 or outer_aspect > 1.8:
                continue

            inner_points = np.vstack(hole_contours)
            opening_bbox = cv2.boundingRect(inner_points)
            ix, iy, iw, ih = opening_bbox
            hole_area = float(sum(cv2.contourArea(h) for h in hole_contours))
            hole_ratio = hole_area / max(outer_area, 1.0)
            opening_aspect = iw / max(ih, 1)
            centeredness = 1.0 - (
                abs((ix + 0.5 * iw) - (outer_x + 0.5 * outer_w)) / max(outer_w, 1)
                + abs((iy + 0.5 * ih) - (outer_y + 0.5 * outer_h)) / max(outer_h, 1)
            )
            centeredness = max(0.0, centeredness)

            score = (
                0.35 * max(0.0, 1.0 - abs(opening_aspect - 1.0) / 0.55)
                + 0.35 * min(1.0, hole_ratio / 0.45)
                + 0.20 * centeredness
                + 0.10 * min(1.0, (outer_w * outer_h) / max(0.12 * W * H, 1.0))
            )

            if score <= best_score:
                continue

            rect = cv2.minAreaRect(inner_points)
            corners = cv2.boxPoints(rect).astype(np.float32)
            best_score = float(score)
            best_det = GateDetection(
                is_valid=score >= self.config.min_gate_confidence,
                confidence=float(score),
                mode="ring",
                image_size=(W, H),
                center_px=(ix + 0.5 * iw, iy + 0.5 * ih),
                outer_bbox=outer_bbox,
                opening_bbox=opening_bbox,
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

    def _detect_post_pair_hypothesis(self, mask: np.ndarray, W: int, H: int) -> Optional[GateDetection]:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        verticals: list[dict[str, Any]] = []
        horizontals: list[dict[str, Any]] = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if area < self.config.min_component_area:
                continue

            aspect = h / max(w, 1)
            item = {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": float(centroids[i][0]),
                "cy": float(centroids[i][1]),
                "aspect": float(aspect),
            }
            if aspect >= self.config.min_vertical_aspect and h >= self.config.min_vertical_height_ratio * H:
                verticals.append(item)
            if (w / max(h, 1)) >= 1.35 and w >= 0.08 * W:
                horizontals.append(item)

        verticals.sort(key=lambda c: c["cx"])
        best_det = None
        best_score = 0.0

        for i in range(len(verticals)):
            for j in range(i + 1, len(verticals)):
                left = verticals[i]
                right = verticals[j]

                gap = right["x"] - (left["x"] + left["w"])
                if gap < 18 or gap > 0.78 * W:
                    continue

                vertical_overlap = max(
                    0,
                    min(left["y"] + left["h"], right["y"] + right["h"]) - max(left["y"], right["y"]),
                )
                overlap_ratio = vertical_overlap / max(min(left["h"], right["h"]), 1)
                if overlap_ratio < 0.40:
                    continue

                height_ratio = min(left["h"], right["h"]) / max(left["h"], right["h"])
                width_ratio = min(left["w"], right["w"]) / max(left["w"], right["w"])
                gap_ratio = gap / max(0.5 * (left["h"] + right["h"]), 1.0)
                gap_score = max(0.0, 1.0 - abs(gap_ratio - 0.72) / 0.85)

                top_y = int(min(left["y"], right["y"]))
                bottom_y = int(max(left["y"] + left["h"], right["y"] + right["h"]))
                x_outer_1 = int(left["x"])
                x_outer_2 = int(right["x"] + right["w"])
                x_inner_1 = int(left["x"] + left["w"])
                x_inner_2 = int(right["x"])
                if x_inner_2 <= x_inner_1:
                    continue

                band = max(4, int(0.06 * max(left["h"], right["h"])))
                top_band_y1 = max(0, top_y - band)
                top_band_y2 = min(H, top_y + band)
                top_band = mask[top_band_y1:top_band_y2, x_outer_1:x_outer_2]
                top_occupancy = float(top_band.mean() / 255.0) if top_band.size else 0.0

                inner_y1 = min(H, top_y + band)
                inner_y2 = max(inner_y1 + 1, min(H, bottom_y - band))
                opening_band = mask[inner_y1:inner_y2, x_inner_1:x_inner_2]
                opening_occupancy = float(opening_band.mean() / 255.0) if opening_band.size else 1.0
                hollow_score = max(0.0, 1.0 - opening_occupancy / 0.18)

                horizontal_support = 0.0
                for hcomp in horizontals:
                    spans_pair = hcomp["x"] <= x_inner_1 and (hcomp["x"] + hcomp["w"]) >= x_inner_2
                    near_top = abs(hcomp["y"] - top_y) <= 0.20 * max(left["h"], right["h"])
                    if spans_pair and near_top:
                        horizontal_support = max(horizontal_support, min(1.0, hcomp["w"] / max(x_outer_2 - x_outer_1, 1)))

                symmetry_score = 0.6 * height_ratio + 0.4 * width_ratio
                top_support_score = min(1.0, 0.65 * top_occupancy + 0.35 * horizontal_support)
                edge_penalty = 0.0
                if left["x"] < 4 or right["x"] + right["w"] > W - 4:
                    edge_penalty += 0.12
                if (x_outer_2 - x_outer_1) < 0.12 * W:
                    edge_penalty += 0.10

                score = (
                    0.24 * symmetry_score
                    + 0.22 * overlap_ratio
                    + 0.22 * gap_score
                    + 0.20 * top_support_score
                    + 0.12 * hollow_score
                    - edge_penalty
                )

                if score <= best_score:
                    continue

                outer_bbox = (x_outer_1, top_y, x_outer_2 - x_outer_1, bottom_y - top_y)
                opening_bbox = (x_inner_1, top_y, x_inner_2 - x_inner_1, bottom_y - top_y)
                cx = x_inner_1 + 0.5 * (x_inner_2 - x_inner_1)
                cy = top_y + 0.5 * (bottom_y - top_y)
                corners = np.array(
                    [
                        [x_inner_1, top_y],
                        [x_inner_2, top_y],
                        [x_inner_2, bottom_y],
                        [x_inner_1, bottom_y],
                    ],
                    dtype=np.float32,
                )

                best_score = float(score)
                best_det = GateDetection(
                    is_valid=score >= self.config.min_gate_confidence,
                    confidence=float(score),
                    mode="post_pair",
                    image_size=(W, H),
                    center_px=(float(cx), float(cy)),
                    outer_bbox=outer_bbox,
                    opening_bbox=opening_bbox,
                    corners_px=corners,
                    opening_width_px=float(opening_bbox[2]),
                    opening_height_px=float(opening_bbox[3]),
                    debug={
                        "left": left,
                        "right": right,
                        "height_ratio": height_ratio,
                        "width_ratio": width_ratio,
                        "overlap_ratio": overlap_ratio,
                        "gap_ratio": gap_ratio,
                        "top_occupancy": top_occupancy,
                        "opening_occupancy": opening_occupancy,
                        "horizontal_support": horizontal_support,
                        "symmetry_score": symmetry_score,
                        "top_support_score": top_support_score,
                        "hollow_score": hollow_score,
                    },
                )

        return best_det

    def _add_control_quantities(self, det: GateDetection) -> GateDetection:
        W, H = det.image_size
        if det.center_px is not None:
            cx, cy = det.center_px
            det.lateral_error_norm = float((cx - 0.5 * W) / max(0.5 * W, 1.0))
            det.vertical_error_norm = float((cy - 0.5 * H) / max(0.5 * H, 1.0))

        if det.opening_bbox is not None:
            x, y, w, h = det.opening_bbox
            det.opening_width_px = float(w)
            det.opening_height_px = float(h)

        if det.mode == "post_pair" and "left" in det.debug and "right" in det.debug:
            left = det.debug["left"]
            right = det.debug["right"]
            width_asym = (right["w"] - left["w"]) / max(right["w"] + left["w"], 1)
            height_asym = (right["h"] - left["h"]) / max(right["h"] + left["h"], 1)
            det.yaw_proxy = float(0.65 * width_asym + 0.35 * height_asym)
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

    def _can_estimate_pose(self, det: GateDetection) -> bool:
        return (
            det.corners_px is not None
            and self.config.gate_opening_width_m is not None
            and self.config.gate_opening_height_m is not None
            and self.config.camera_matrix is not None
            and self.config.dist_coeffs is not None
        )

    def _estimate_pose(self, det: GateDetection) -> None:
        half_w = 0.5 * float(self.config.gate_opening_width_m)
        half_h = 0.5 * float(self.config.gate_opening_height_m)
        object_points = np.array(
            [
                [-half_w, -half_h, 0.0],
                [half_w, -half_h, 0.0],
                [half_w, half_h, 0.0],
                [-half_w, half_h, 0.0],
            ],
            dtype=np.float32,
        )
        image_points = self._order_corners(det.corners_px.astype(np.float32))
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.config.camera_matrix,
            self.config.dist_coeffs,
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


def load_image_paths(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the enhanced gate detector on an image folder.")
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--gate-width-m", type=float, default=None)
    parser.add_argument("--fx-px", type=float, default=None)
    args = parser.parse_args()

    cfg = GateDetectorConfig(
        min_gate_confidence=args.min_confidence,
        gate_opening_width_m=args.gate_width_m,
        fx_px=args.fx_px,
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
        print(f"{path.name}\tvalid={det.is_valid}\tconf={det.confidence:.3f}\tmode={det.mode}\tcenter={det.center_px}")

    print(f"Detected {valid_count}/{len(paths)} valid frames")
