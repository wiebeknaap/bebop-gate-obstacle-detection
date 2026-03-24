import cv2
import numpy as np
import os


DATASET_PATH = "data/raw"
OUTPUT_PATH  = "data/output_edge_frames"

os.makedirs(OUTPUT_PATH, exist_ok=True)

EDGE_DENSITY_THRESHOLD = 0.1

ROI_X_START = 0.1
ROI_X_END   = 0.9
ROI_Y_START = 0.10
ROI_Y_END   = 0.90

GRID_ROWS = 40
GRID_COLS = 30

SOBEL_KSIZE     = 3
SOBEL_THRESHOLD = 30

MIN_VERTICAL_FILL = 0.03
TURN_THRESHOLD = 0.4


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


def find_safe_col(cell_obstacle):
    mid_row = GRID_ROWS // 2
    mid_col = GRID_COLS // 2
    middle_rows = cell_obstacle[mid_row - 1 : mid_row + 2, :]  # 3 middle rows

    # a column is safe if no obstacle in any of the 3 middle rows
    col_safe = [not middle_rows[:, c].any() for c in range(GRID_COLS)]

    # find 2 consecutive safe columns, closest to center
    order = sorted(range(GRID_COLS - 1), key=lambda c: abs(c - mid_col))
    for c in order:
        if col_safe[c] and col_safe[c + 1]:
            return c  # return left column of the safe pair

    return None


def analyze_front(cell_obstacle):
    mid_col = GRID_COLS // 2

    # all rows, middle 5 columns
    grid = cell_obstacle[:, mid_col - 2 : mid_col + 3]

    total        = grid.size
    unsafe_count = grid.sum()
    unsafe_ratio = unsafe_count / total

    if unsafe_ratio >= 0.7:
        proximity = "CLOSE"
    elif unsafe_ratio >= 0.4:
        proximity = "MEDIUM"
    else:
        proximity = "FAR"

    return unsafe_ratio, proximity


def build_vis(frame, roi_coords, edge_mask, cell_density, cell_obstacle,
              safe_col, obstacle):
    h, w = frame.shape[:2]
    roi_coords = (0, 0, w, h)
    x0, y0, x1, y1 = roi_coords
    roi_h = y1 - y0
    roi_w = x1 - x0
    cell_h = roi_h // GRID_ROWS
    cell_w = roi_w // GRID_COLS

    vis = frame.copy()

    mid_row = GRID_ROWS // 2
    mid_col = GRID_COLS // 2

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = x1 if c == GRID_COLS - 1 else cx0 + cell_w
            cy1 = y1 if r == GRID_ROWS - 1 else cy0 + cell_h

            # highlight the center 5x5 grid with a border
            in_center = (mid_col - 2 <= c <= mid_col + 2)

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

            if in_center:
                cv2.rectangle(vis, (cx0, cy0), (cx1, cy1), (255, 255, 0), 2)

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
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi, roi_coords = get_roi(gray)

    edge_mask                   = vertical_edges_sobel(gray_roi)
    edge_mask                   = filter_by_vertical_fill(edge_mask)
    cell_density, cell_obstacle = compute_grid(edge_mask)

    safe_col                    = find_safe_col(cell_obstacle)
    obstacle                    = cell_obstacle.any()
    unsafe_ratio, proximity     = analyze_front(cell_obstacle)

    mid_col = GRID_COLS // 2
    if unsafe_ratio >= TURN_THRESHOLD:
        if safe_col is None or safe_col == mid_col:
            direction = "TURN RIGHT"  # default when no clear path
        elif safe_col < mid_col:
            direction = "TURN LEFT"
        else:
            direction = "TURN RIGHT"
    else:
        direction = "GO STRAIGHT"

    print(f"{img_name}: proximity={proximity} ({unsafe_ratio:.0%} unsafe) | "
          f"safe_col={safe_col} → {direction}")
    print(f"  Obstacle counts per column: {cell_obstacle.sum(axis=0).tolist()}")

    vis = build_vis(frame, roi_coords, edge_mask,
                    cell_density, cell_obstacle, safe_col, obstacle)

    cv2.imwrite(os.path.join(OUTPUT_PATH, img_name), vis)

print("-" * 60)
print(f"Done. Saved to: {OUTPUT_PATH}")