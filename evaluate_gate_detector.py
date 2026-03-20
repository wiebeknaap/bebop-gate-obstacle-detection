from __future__ import annotations

from pathlib import Path
import argparse
import math

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enhanced_gate_detector import EnhancedGateDetector, GateDetectorConfig, load_image_paths


def build_montage(image_paths: list[Path], output_path: Path, title: str, ncols: int = 4) -> None:
    n = len(image_paths)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, path in zip(axes.ravel(), image_paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(path.name, fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the enhanced gate detector on a folder of frames.")
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("gate_eval_output"))
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--max-save", type=int, default=24)
    parser.add_argument("--gate-width-m", type=float, default=None)
    parser.add_argument("--fx-px", type=float, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    detector = EnhancedGateDetector(
        GateDetectorConfig(
            min_gate_confidence=args.min_confidence,
            gate_opening_width_m=args.gate_width_m,
            fx_px=args.fx_px,
        )
    )

    rows = []
    overlay_paths = []
    for path in load_image_paths(args.image_folder):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        proc, mask, det = detector.detect(frame)
        annotated = detector.annotate(proc, det, mask)
        if len(overlay_paths) < args.max_save:
            out_path = overlay_dir / path.name
            cv2.imwrite(str(out_path), annotated)
            overlay_paths.append(out_path)

        rows.append(
            {
                "frame": path.name,
                "valid": det.is_valid,
                "confidence": det.confidence,
                "mode": det.mode,
                "center_x": None if det.center_px is None else det.center_px[0],
                "center_y": None if det.center_px is None else det.center_px[1],
                "lateral_error_norm": det.lateral_error_norm,
                "vertical_error_norm": det.vertical_error_norm,
                "yaw_proxy": det.yaw_proxy,
                "opening_width_px": det.opening_width_px,
                "opening_height_px": det.opening_height_px,
                "range_estimate_m": det.range_estimate_m,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "gate_detection_summary.csv"
    df.to_csv(csv_path, index=False)

    if overlay_paths:
        build_montage(
            overlay_paths,
            args.output_dir / "gate_detection_montage.png",
            "Enhanced gate detector: first saved overlays",
        )

    valid_rate = float(df["valid"].mean()) if len(df) else 0.0
    conf_mean = float(df["confidence"].mean()) if len(df) else 0.0
    print(f"Frames processed: {len(df)}")
    print(f"Valid detection rate: {valid_rate:.3f}")
    print(f"Mean confidence: {conf_mean:.3f}")
    print(f"Summary CSV: {csv_path}")
