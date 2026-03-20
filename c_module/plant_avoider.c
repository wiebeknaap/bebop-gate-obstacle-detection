#include "plant_avoider.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include <string.h>
#include <stdlib.h>

PlantAvoiderResult plant_avoider_result;

static uint8_t get_y(const struct image_t *img, int x, int y);
static void    run_detection(const struct image_t *img);

void plant_avoider_init(void)
{
    memset(&plant_avoider_result, 0, sizeof(plant_avoider_result));
    cv_add_to_device(&PLANT_AVOIDER_CAMERA, plant_avoider_detect, PLANT_AVOIDER_FPS, 0);
}

void plant_avoider_periodic(void)
{
}

struct image_t *plant_avoider_detect(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
    run_detection(img);
    return NULL;
}

static void run_detection(const struct image_t *img)
{
    const int W = img->w;
    const int H = img->h;

    const int x0 = (int)(ROI_X_START * W);
    const int x1 = (int)(ROI_X_END   * W);
    const int y0 = (int)(ROI_Y_START * H);
    const int y1 = (int)(ROI_Y_END   * H);

    const int roi_w = x1 - x0;
    const int roi_h = y1 - y0;

    const int cell_w = roi_w / GRID_COLS;
    const int cell_h = roi_h / GRID_ROWS;

    if (cell_w <= 0 || cell_h <= 0) return;

    int col_edge_count[roi_w];
    memset(col_edge_count, 0, sizeof(int) * roi_w);

    for (int ry = 1; ry < roi_h - 1; ry++) {
        int gy = y0 + ry;
        for (int rx = 1; rx < roi_w - 1; rx++) {
            int gx = x0 + rx;

            int sx = -1 * get_y(img, gx-1, gy-1) + 1 * get_y(img, gx+1, gy-1)
                   + -2 * get_y(img, gx-1, gy)   + 2 * get_y(img, gx+1, gy)
                   + -1 * get_y(img, gx-1, gy+1) + 1 * get_y(img, gx+1, gy+1);

            if (sx < 0) sx = -sx;

            if (sx > SOBEL_THRESHOLD) {
                col_edge_count[rx]++;
            }
        }
    }

    const int min_pixels = (int)(MIN_VERTICAL_FILL * roi_h);
    int col_valid[roi_w];
    for (int rx = 0; rx < roi_w; rx++) {
        col_valid[rx] = (col_edge_count[rx] >= min_pixels) ? 1 : 0;
    }

    float cell_density [GRID_ROWS][GRID_COLS];
    int   cell_obstacle[GRID_ROWS][GRID_COLS];

    for (int r = 0; r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {

            int cx0 = c * cell_w;
            int cy0 = y0 + r * cell_h;
            int cy1 = cy0 + cell_h;

            int edge_count = 0;
            int total      = 0;

            for (int ry = cy0; ry < cy1; ry++) {
                int gy = ry;
                for (int rx = cx0; rx < cx0 + cell_w; rx++) {
                    int gx = x0 + rx;

                    if (!col_valid[rx]) { total++; continue; }

                    int sx = -1 * get_y(img, gx-1, gy-1) + 1 * get_y(img, gx+1, gy-1)
                           + -2 * get_y(img, gx-1, gy)   + 2 * get_y(img, gx+1, gy)
                           + -1 * get_y(img, gx-1, gy+1) + 1 * get_y(img, gx+1, gy+1);
                    if (sx < 0) sx = -sx;

                    if (sx > SOBEL_THRESHOLD) edge_count++;
                    total++;
                }
            }

            float density = (total > 0) ? (float)edge_count / (float)total : 0.f;
            cell_density [r][c] = density;
            cell_obstacle[r][c] = (density > EDGE_DENSITY_THRESHOLD) ? 1 : 0;
        }
    }

    int col_obstacle_count[GRID_COLS];
    for (int c = 0; c < GRID_COLS; c++) {
        col_obstacle_count[c] = 0;
        for (int r = 0; r < GRID_ROWS; r++) {
            col_obstacle_count[c] += cell_obstacle[r][c];
        }
    }

    int safe_col   = 1;
    int safe_count = col_obstacle_count[0] + col_obstacle_count[1] + col_obstacle_count[2];
    for (int c = 1; c < GRID_COLS - 1; c++) {
        int window = col_obstacle_count[c-1] + col_obstacle_count[c] + col_obstacle_count[c+1];
        if (window < safe_count) {
            safe_count = window;
            safe_col   = c;
        }
    }

    int obstacle  = 0;
    int mid_start = GRID_ROWS / 2 - 1;
    int mid_end   = GRID_ROWS / 2 + 2;
    for (int r = mid_start; r < mid_end && r < GRID_ROWS; r++) {
        for (int c = 0; c < GRID_COLS; c++) {
            if (cell_obstacle[r][c]) { obstacle = 1; break; }
        }
        if (obstacle) break;
    }

    static float smooth_safe_col = 4.5f;
    smooth_safe_col = 0.7f * smooth_safe_col + 0.3f * (float)safe_col;

    static int obstacle_count = 0;
    if (obstacle) { obstacle_count++; } else { obstacle_count--; }
    if (obstacle_count < 0) obstacle_count = 0;
    if (obstacle_count > 5) obstacle_count = 5;

    plant_avoider_result.safe_col          = (int)(smooth_safe_col + 0.5f);
    plant_avoider_result.obstacle_detected = (obstacle_count >= 3) ? 1 : 0;
    memcpy(plant_avoider_result.cell_density,  cell_density,  sizeof(cell_density));
    memcpy(plant_avoider_result.cell_obstacle, cell_obstacle, sizeof(cell_obstacle));
}

static uint8_t get_y(const struct image_t *img, int x, int y)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img->w) x = img->w - 1;
    if (y >= img->h) y = img->h - 1;

    const uint8_t *buf = (const uint8_t *)img->buf;
    return buf[(y * img->w + x) * 2 + 1];
}