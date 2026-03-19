import cv2
import numpy as np
import os


DATASET_PATH = "data/raw"
OUTPUT_PATH  = "data/output_edge_frames"

os.makedirs(OUTPUT_PATH, exist_ok=True)


EDGE_DENSITY_THRESHOLD = 0.10

ROI_X_START = 0.1
ROI_X_END   = 0.9
ROI_Y_START = 0.10
ROI_Y_END   = 0.90

GRID_ROWS = 25
GRID_COLS = 20

SOBEL_KSIZE     = 3
SOBEL_THRESHOLD = 30

MIN_VERTICAL_FILL = 0.03


def get_roi(frame):
    h, w = frame.shape[:2]
    x0 = int(ROI_X_START * w)
    x1 = int(ROI_X_END   * w)
    y0 = int(ROI_Y_START * h)
    y1 = int(ROI_Y_END   * h)
    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def vertical_edges_sobel(gray_roi):
    sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
    sobel_x = np.abs(sobel_x)
    mask = (sobel_x > SOBEL_THRESHOLD).astype(np.uint8) * 255
    return mask


def filter_by_vertical_fill(mask):
    h = mask.shape[0]
    min_pixels = int(MIN_VERTICAL_FILL * h)
    col_sums   = np.sum(mask > 0, axis=0)
    valid_cols = col_sums >= min_pixels
    filtered   = mask.copy()
    filtered[:, ~valid_cols] = 0
    return filtered


def compute_grid(edge_mask):
    h, w = edge_mask.shape
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    cell_density  = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
    cell_obstacle = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0 = r * cell_h
            y1 = y0 + cell_h
            x0 = c * cell_w
            x1 = x0 + cell_w
            cell = edge_mask[y0:y1, x0:x1]
            d = np.count_nonzero(cell) / cell.size
            cell_density[r, c]  = d
            cell_obstacle[r, c] = d > EDGE_DENSITY_THRESHOLD

    return cell_density, cell_obstacle


def find_safe_column(cell_obstacle):
    col_obstacle_count = cell_obstacle.sum(axis=0)
    best_col = int(np.argmin(col_obstacle_count))
    return best_col, col_obstacle_count


def obstacle_decision(cell_obstacle):
    middle = cell_obstacle[3:7, :]   # middle 4 rows of 10
    return bool(middle.any())


def build_vis(frame, roi_coords, edge_mask, cell_density, cell_obstacle,
              safe_col, obstacle):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi_coords
    roi_h = y1 - y0
    roi_w = x1 - x0
    cell_h = roi_h // GRID_ROWS
    cell_w = roi_w // GRID_COLS

    vis = frame.copy()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = cx0 + cell_w
            cy1 = cy0 + cell_h

            if cell_obstacle[r, c]:
                overlay = vis.copy()
                cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.rectangle(vis, (cx0, cy0), (cx1, cy1), (0, 0, 255), 2)
            else:
                overlay = vis.copy()
                cv2.rectangle(overlay, (cx0, cy0), (cx1, cy1), (0, 180, 0), -1)
                cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                cv2.rectangle(vis, (cx0, cy0), (cx1, cy1), (0, 255, 0), 2)

            cv2.putText(vis, f"{cell_density[r,c]:.2f}",
                        (cx0 + 4, cy0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1)

    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)

    return vis

SELECTED_IMAGES = {
    "812255641.jpg",
    "815322282.jpg",
    "815422288.jpg",
    "817888933.jpg",
    "823488936.jpg",
}

images = sorted([
    f for f in os.listdir(DATASET_PATH)
    if f in SELECTED_IMAGES
])


for img_name in images:
    frame = cv2.imread(os.path.join(DATASET_PATH, img_name))
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi, roi_coords = get_roi(gray)

    edge_mask                   = vertical_edges_sobel(gray_roi)
    edge_mask                   = filter_by_vertical_fill(edge_mask)
    cell_density, cell_obstacle = compute_grid(edge_mask)
    safe_col, _                 = find_safe_column(cell_obstacle)
    obstacle                    = obstacle_decision(cell_obstacle)


    vis = build_vis(frame, roi_coords, edge_mask,
                    cell_density, cell_obstacle, safe_col, obstacle)

    cv2.imwrite(os.path.join(OUTPUT_PATH, img_name), vis)

print("-" * 60)
print(f"Done. Saved to: {OUTPUT_PATH}")