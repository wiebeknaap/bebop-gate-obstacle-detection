from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class GateDetectorConfig:
    rotate_portrait_frames: bool = True

    min_component_area: int = 120
    min_vertical_aspect: float = 1.0
    min_vertical_height_ratio: float = 0.05

    min_gate_confidence: float = 0.50

    # Ring detection is kept optional, but disabled by default because in this
    # dataset the gate is much more reliably characterized by two blue posts
    # than by a fully blue ring.
    enable_ring_hypothesis: bool = False
    strong_ring_confidence: float = 0.76
    ring_selection_margin: float = 0.12

    hsv_lower_blue: tuple[int, int, int] = (85, 35, 18)
    hsv_upper_blue: tuple[int, int, int] = (145, 255, 255)

    min_bg_diff: int = 18
    min_br_diff: int = 12
    min_saturation_for_rgb_mask: int = 30
    blue_ratio_min: float = 0.34

    enable_clahe: bool = True
    clahe_clip_limit: float = 2.2
    clahe_tile_grid: tuple[int, int] = (8, 8)

    open_kernel_size: int = 3
    close_kernel_size: int = 5
    vertical_close_kernel_h: int = 11
    horizontal_close_kernel_w: int = 11

    tracker_measurement_confidence: float = 0.28

    min_opening_width_ratio: float = 0.08
    min_opening_height_ratio: float = 0.10
    min_top_support_score: float = 0.06
    min_hollow_score: float = 0.12

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

    Important: tracker output is used only as a non-valid fallback estimate.
    It should never inflate offline detection metrics by itself.
    """

    def __init__(self, alpha: float = 0.70, beta: float = 0.22, confidence_decay: float = 0.92):
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
        has_measurement = (
            detection.opening_bbox is not None
            and (detection.is_valid or detection.confidence >= accept_confidence)
        )

        if not self.state.valid:
            if has_measurement:
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

        if not has_measurement:
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

        self.state.confidence = max(
            self.state.confidence * self.confidence_decay,
            float(detection.confidence),
        )
        return self.state


class EnhancedGateDetector:
    """Blue-first, geometry-aware gate detector."""

    def __init__(self, config: Optional[GateDetectorConfig] = None):
        self.config = config or GateDetectorConfig()
        self.tracker = AlphaBetaGateTracker(alpha=0.70, beta=0.22, confidence_decay=0.92)

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
            lab = cv2.merge((l, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame

    def segment_gate_blue(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(frame_bgr)

        b_i = b.astype(np.int16)
        g_i = g.astype(np.int16)
        r_i = r.astype(np.int16)

        sum_rgb = b.astype(np.float32) + g.astype(np.float32) + r.astype(np.float32) + 1.0
        blue_ratio = b.astype(np.float32) / sum_rgb

        mask_hsv = cv2.inRange(
            hsv,
            np.array(self.config.hsv_lower_blue, dtype=np.uint8),
            np.array(self.config.hsv_upper_blue, dtype=np.uint8),
        )

        mask_rgb = (
            (b_i - g_i > self.config.min_bg_diff)
            & (b_i - r_i > self.config.min_br_diff)
            & (hsv[:, :, 1] > self.config.min_saturation_for_rgb_mask)
        ).astype(np.uint8) * 255

        mask_ratio = (blue_ratio > self.config.blue_ratio_min).astype(np.uint8) * 255

        mask = cv2.bitwise_and(mask_hsv, cv2.bitwise_or(mask_rgb, mask_ratio))

        k_open = np.ones((self.config.open_kernel_size, self.config.open_kernel_size), np.uint8)
        k_close = np.ones((self.config.close_kernel_size, self.config.close_kernel_size), np.uint8)
        k_vclose = np.ones((self.config.vertical_close_kernel_h, 3), np.uint8)
        k_hclose = np.ones((3, self.config.horizontal_close_kernel_w), np.uint8)

        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_vclose)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_hclose)

        # Suppress a tiny bottom strip to reduce floor-mat clutter.
        h, _ = mask.shape
        mask[int(0.95 * h):, :] = 0
        return mask

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, GateDetection]:
        frame = self.preprocess(frame_bgr)
        mask = self.segment_gate_blue(frame)
        image_h, image_w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)

        ring_candidate = None
        if self.config.enable_ring_hypothesis:
            ring_candidate = self._detect_ring_hypothesis(mask, image_w, image_h)

        post_candidate = self._detect_post_pair_hypothesis(mask, edges, image_w, image_h)
        selected = self._select_candidate(ring_candidate, post_candidate)

        if selected is not None:
            selected = self._add_control_quantities(selected)

            self.tracker.update(
                selected,
                accept_confidence=self.config.tracker_measurement_confidence,
            )

            if self._can_estimate_pose(selected):
                self._estimate_pose(selected)

            return frame, mask, selected

        self.tracker.predict(1 / 30.0)
        tracked = self._tracker_to_detection((image_w, image_h))
        if tracked is not None:
            return frame, mask, tracked

        return frame, mask, GateDetection(
            is_valid=False,
            confidence=0.0,
            mode="none",
            image_size=(image_w, image_h),
            debug={"reason": "no hypothesis survived"},
        )

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detection: GateDetection,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        if mask is not None:
            mask_small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
            mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            vis[0:mask_small.shape[0], 0:mask_small.shape[1]] = mask_small

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

        status = f"{detection.mode} conf={detection.confidence:.2f}"
        if detection.lateral_error_norm is not None and detection.vertical_error_norm is not None:
            status += f" ex={detection.lateral_error_norm:+.2f} ey={detection.vertical_error_norm:+.2f}"
        if detection.range_estimate_m is not None:
            status += f" range~{detection.range_estimate_m:.2f}m"

        cv2.putText(vis, status, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return vis

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

        if ring_candidate is not None and ring_candidate.is_valid and ring_candidate.confidence >= self.config.strong_ring_confidence:
            return ring_candidate

        candidates = [c for c in (post_candidate, ring_candidate) if c is not None]
        if not candidates:
            return None

        if post_candidate is not None and ring_candidate is not None:
            if post_candidate.confidence >= ring_candidate.confidence - self.config.ring_selection_margin:
                return post_candidate

        return max(candidates, key=lambda d: d.confidence)

    def _merge_nearby_verticals(self, comps: list[dict[str, Any]], image_w: int) -> list[dict[str, Any]]:
        if not comps:
            return comps

        comps = sorted(comps, key=lambda c: c["x"])
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
                area = prev["area"] + cur["area"]
                bw = x2 - x1
                bh = y2 - y1

                merged[-1] = {
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(bw),
                    "h": int(bh),
                    "area": int(area),
                    "cx": float(x1 + 0.5 * bw),
                    "cy": float(y1 + 0.5 * bh),
                    "aspect": float(bh / max(bw, 1)),
                }
            else:
                merged.append(cur)

        return merged

    def _tracker_to_detection(self, image_size: tuple[int, int]) -> Optional[GateDetection]:
        st = self.tracker.state
        image_w, image_h = image_size

        if not st.valid or st.confidence < self.config.min_gate_confidence:
            return None
        if st.width < 8 or st.height < 8:
            return None

        x = int(round(st.center_x - 0.5 * st.width))
        y = int(round(st.center_y - 0.5 * st.height))
        bw = int(round(st.width))
        bh = int(round(st.height))

        x = max(0, min(image_w - 1, x))
        y = max(0, min(image_h - 1, y))
        bw = max(1, min(image_w - x, bw))
        bh = max(1, min(image_h - y, bh))

        det = GateDetection(
            is_valid=False,
            confidence=float(min(st.confidence, self.config.min_gate_confidence - 1e-3)),
            mode="tracked",
            image_size=(image_w, image_h),
            center_px=(float(st.center_x), float(st.center_y)),
            outer_bbox=(x, y, bw, bh),
            opening_bbox=(x, y, bw, bh),
            opening_width_px=float(bw),
            opening_height_px=float(bh),
            debug={"source": "tracker"},
        )
        return self._add_control_quantities(det)

    def _detect_ring_hypothesis(self, mask: np.ndarray, image_w: int, image_h: int) -> Optional[GateDetection]:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return None

        hierarchy = hierarchy[0]
        best_det = None
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
                area = cv2.contourArea(contours[j])
                if area > 650:
                    hole_contours.append(contours[j])
                j = hierarchy[j][0]

            if not hole_contours:
                continue

            outer_bbox = cv2.boundingRect(contour)
            outer_x, outer_y, outer_w, outer_h = outer_bbox
            outer_aspect = outer_w / max(outer_h, 1)
            if outer_aspect < 0.65 or outer_aspect > 1.60:
                continue

            inner_points = np.vstack(hole_contours)
            opening_bbox = cv2.boundingRect(inner_points)
            ix, iy, iw, ih = opening_bbox

            if iw < max(16, int(self.config.min_opening_width_ratio * image_w)):
                continue
            if ih < max(20, int(self.config.min_opening_height_ratio * image_h)):
                continue

            hole_area = float(sum(cv2.contourArea(h) for h in hole_contours))
            hole_ratio = hole_area / max(outer_area, 1.0)
            opening_aspect = iw / max(ih, 1)

            centeredness = 1.0 - (
                abs((ix + 0.5 * iw) - (outer_x + 0.5 * outer_w)) / max(outer_w, 1)
                + abs((iy + 0.5 * ih) - (outer_y + 0.5 * outer_h)) / max(outer_h, 1)
            )
            centeredness = max(0.0, centeredness)
            if centeredness < 0.22:
                continue

            aspect_score = max(0.0, 1.0 - abs(opening_aspect - 0.95) / 0.65)
            hole_ratio_score = max(0.0, 1.0 - abs(hole_ratio - 0.30) / 0.28)
            size_score = min(1.0, (outer_w * outer_h) / max(0.12 * image_w * image_h, 1.0))

            score = (
                0.33 * aspect_score
                + 0.29 * hole_ratio_score
                + 0.23 * centeredness
                + 0.15 * size_score
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
                image_size=(image_w, image_h),
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
                    "aspect_score": aspect_score,
                    "hole_ratio_score": hole_ratio_score,
                    "size_score": size_score,
                },
            )

        return best_det

    def _detect_post_pair_hypothesis(
        self,
        mask: np.ndarray,
        edges: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> Optional[GateDetection]:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        verticals: list[dict[str, Any]] = []

        for i in range(1, n_labels):
            x, y, bw, bh, area = stats[i]
            if area < self.config.min_component_area:
                continue

            aspect = bh / max(bw, 1)
            if aspect < self.config.min_vertical_aspect:
                continue
            if bh < self.config.min_vertical_height_ratio * image_h:
                continue

            fill = float(mask[y:y + bh, x:x + bw].mean() / 255.0) if bw > 0 and bh > 0 else 0.0
            if fill < 0.16:
                continue

            verticals.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(bw),
                    "h": int(bh),
                    "area": int(area),
                    "cx": float(centroids[i][0]),
                    "cy": float(centroids[i][1]),
                    "aspect": float(aspect),
                    "fill": fill,
                }
            )

        verticals = self._merge_nearby_verticals(verticals, image_w)
        verticals.sort(key=lambda c: c["cx"])

        best_det = None
        best_score = 0.0

        min_opening_width_px = max(20, int(self.config.min_opening_width_ratio * image_w))
        min_opening_height_px = max(22, int(self.config.min_opening_height_ratio * image_h))

        for i in range(len(verticals)):
            for j in range(i + 1, len(verticals)):
                left = verticals[i]
                right = verticals[j]

                gap = right["x"] - (left["x"] + left["w"])
                if gap < 12 or gap > 0.82 * image_w:
                    continue

                vertical_overlap = max(
                    0,
                    min(left["y"] + left["h"], right["y"] + right["h"]) - max(left["y"], right["y"]),
                )
                overlap_ratio = vertical_overlap / max(min(left["h"], right["h"]), 1)
                if overlap_ratio < 0.28:
                    continue

                height_ratio = min(left["h"], right["h"]) / max(left["h"], right["h"])
                width_ratio = min(left["w"], right["w"]) / max(left["w"], right["w"])
                if height_ratio < 0.45:
                    continue
                if width_ratio < 0.15:
                    continue

                top_y = int(min(left["y"], right["y"]))
                bottom_y = int(max(left["y"] + left["h"], right["y"] + right["h"]))
                x_outer_1 = int(left["x"])
                x_outer_2 = int(right["x"] + right["w"])
                x_inner_1 = int(left["x"] + left["w"])
                x_inner_2 = int(right["x"])

                if x_inner_2 <= x_inner_1:
                    continue

                pair_h = max(1, bottom_y - top_y)
                opening_w = x_inner_2 - x_inner_1
                opening_h = pair_h

                if opening_w < min_opening_width_px:
                    continue
                if opening_h < min_opening_height_px:
                    continue

                opening_aspect = opening_w / max(opening_h, 1)
                opening_aspect_score = max(0.0, 1.0 - abs(opening_aspect - 0.68) / 0.80)

                gap_ratio = gap / max(0.5 * (left["h"] + right["h"]), 1.0)
                gap_score = max(0.0, 1.0 - abs(gap_ratio - 0.70) / 1.20)

                # Top support is computed from image edges, not blue-mask support,
                # because the real gate top is checkered / high-contrast rather than blue.
                band = max(4, int(0.05 * pair_h))
                top_edge_band = edges[max(0, top_y - band):min(image_h, top_y + band), x_outer_1:x_outer_2]
                top_edge_density = float(top_edge_band.mean() / 255.0) if top_edge_band.size else 0.0
                top_support_score = min(1.0, top_edge_density / 0.14)

                margin_x = max(2, int(0.12 * opening_w))
                margin_y = max(2, int(0.08 * opening_h))
                core_x1 = min(image_w, x_inner_1 + margin_x)
                core_x2 = max(core_x1 + 1, min(image_w, x_inner_2 - margin_x))
                core_y1 = min(image_h, top_y + margin_y)
                core_y2 = max(core_y1 + 1, min(image_h, bottom_y - margin_y))
                opening_core = mask[core_y1:core_y2, core_x1:core_x2]
                opening_occupancy = float(opening_core.mean() / 255.0) if opening_core.size else 1.0
                hollow_score = max(0.0, 1.0 - opening_occupancy / 0.20)

                if opening_occupancy > 0.36:
                    continue

                left_patch = mask[left["y"]:left["y"] + left["h"], left["x"]:left["x"] + left["w"]]
                right_patch = mask[right["y"]:right["y"] + right["h"], right["x"]:right["x"] + right["w"]]
                left_fill = float(left_patch.mean() / 255.0) if left_patch.size else 0.0
                right_fill = float(right_patch.mean() / 255.0) if right_patch.size else 0.0
                post_fill_score = min(1.0, 0.5 * (left_fill + right_fill) / 0.55)

                balance_score = 1.0 - abs(left_fill - right_fill) / max(left_fill + right_fill, 1e-6)
                balance_score = max(0.0, balance_score)

                symmetry_score = 0.6 * height_ratio + 0.4 * width_ratio

                center_x = x_inner_1 + 0.5 * (x_inner_2 - x_inner_1)
                center_score = max(0.0, 1.0 - abs(center_x - 0.5 * image_w) / (0.55 * image_w))

                if top_support_score < self.config.min_top_support_score:
                    continue
                if hollow_score < self.config.min_hollow_score:
                    continue

                edge_penalty = 0.0
                if left["x"] < 4 or right["x"] + right["w"] > image_w - 4:
                    edge_penalty += 0.06
                if (x_outer_2 - x_outer_1) < 0.12 * image_w:
                    edge_penalty += 0.06

                score = (
                    0.22 * symmetry_score
                    + 0.18 * overlap_ratio
                    + 0.18 * gap_score
                    + 0.16 * opening_aspect_score
                    + 0.12 * top_support_score
                    + 0.08 * hollow_score
                    + 0.04 * post_fill_score
                    + 0.02 * balance_score
                    + 0.02 * center_score
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
                    image_size=(image_w, image_h),
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
                        "opening_aspect": opening_aspect,
                        "opening_aspect_score": opening_aspect_score,
                        "top_edge_density": top_edge_density,
                        "top_support_score": top_support_score,
                        "opening_occupancy": opening_occupancy,
                        "hollow_score": hollow_score,
                        "post_fill_score": post_fill_score,
                        "balance_score": balance_score,
                        "center_score": center_score,
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
            det.opening_width_px = float(bw)
            det.opening_height_px = float(bh)

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

        print(
            f"{path.name}\tvalid={det.is_valid}\tconf={det.confidence:.3f}\t"
            f"mode={det.mode}\tcenter={det.center_px}"
        )

    print(f"Detected {valid_count}/{len(paths)} valid frames")
