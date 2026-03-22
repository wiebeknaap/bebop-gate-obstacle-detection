#include "plant_avoider.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

struct plant_avoider_result_t plant_avoider_result = {
  .obstacle_detected = false,
  .safe_col          = GRID_COLS / 2,
  .unsafe_ratio      = 0.f,
  .turn_left         = false
};

static pthread_mutex_t pa_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline uint8_t get_y(const uint8_t *buf, uint32_t x, uint32_t y,
                             uint32_t w)
{
  uint32_t idx = y * w * 2 + (x / 2) * 4 + (x % 2 == 0 ? 1 : 3);
  return buf[idx];
}

static inline int32_t sobel_x_at(const uint8_t *buf,
                                  uint32_t px, uint32_t py,
                                  uint32_t img_w, uint32_t img_h)
{
  if (px == 0 || px >= img_w - 1 || py == 0 || py >= img_h - 1) return 0;

  int32_t g =
    -1 * (int32_t)get_y(buf, px-1, py-1, img_w)
    +1 * (int32_t)get_y(buf, px+1, py-1, img_w)
    -2 * (int32_t)get_y(buf, px-1, py,   img_w)
    +2 * (int32_t)get_y(buf, px+1, py,   img_w)
    -1 * (int32_t)get_y(buf, px-1, py+1, img_w)
    +1 * (int32_t)get_y(buf, px+1, py+1, img_w);

  return abs(g);
}

static struct image_t *plant_avoider_cb(struct image_t *img,
                                        uint8_t cam_id __attribute__((unused)))
{
  if (!img || !img->buf) return img;

  const uint32_t W   = img->w;
  const uint32_t H   = img->h;
  const uint8_t *buf = (const uint8_t *)img->buf;

  uint32_t rx0 = (uint32_t)(ROI_X_START * W);
  uint32_t rx1 = (uint32_t)(ROI_X_END   * W);
  uint32_t ry0 = (uint32_t)(ROI_Y_START * H);
  uint32_t ry1 = (uint32_t)(ROI_Y_END   * H);

  uint32_t roi_w = rx1 - rx0;
  uint32_t roi_h = ry1 - ry0;

  uint8_t *mask = (uint8_t *)calloc(roi_w * roi_h, 1);
  if (!mask) return img;

  for (uint32_t ry = 0; ry < roi_h; ry++) {
    for (uint32_t rx = 0; rx < roi_w; rx++) {
      int32_t g = sobel_x_at(buf, rx0 + rx, ry0 + ry, W, H);
      mask[ry * roi_w + rx] = (g > SOBEL_THRESHOLD) ? 1 : 0;
    }
  }

  uint32_t min_pix = (uint32_t)(MIN_VERTICAL_FILL * roi_h);
  for (uint32_t rx = 0; rx < roi_w; rx++) {
    uint32_t sum = 0;
    for (uint32_t ry = 0; ry < roi_h; ry++) sum += mask[ry * roi_w + rx];
    if (sum < min_pix)
      for (uint32_t ry = 0; ry < roi_h; ry++) mask[ry * roi_w + rx] = 0;
  }

  uint32_t cell_h = roi_h / GRID_ROWS;
  uint32_t cell_w = roi_w / GRID_COLS;

  uint8_t obs[GRID_ROWS][GRID_COLS];
  memset(obs, 0, sizeof(obs));

  for (int r = 0; r < GRID_ROWS; r++) {
    for (int c = 0; c < GRID_COLS; c++) {
      uint32_t y0 = r * cell_h;
      uint32_t y1 = (r == GRID_ROWS-1) ? roi_h : y0 + cell_h;
      uint32_t x0 = c * cell_w;
      uint32_t x1 = (c == GRID_COLS-1) ? roi_w : x0 + cell_w;

      uint32_t cnt   = 0;
      uint32_t total = (y1-y0) * (x1-x0);
      for (uint32_t ry = y0; ry < y1; ry++)
        for (uint32_t rx = x0; rx < x1; rx++)
          cnt += mask[ry * roi_w + rx];

      float density = (total > 0) ? (float)cnt / (float)total : 0.f;
      obs[r][c] = (density > EDGE_DENSITY_THRESHOLD) ? 1 : 0;
    }
  }

  free(mask);

  int mid_col = GRID_COLS / 2;
  uint32_t unsafe_cnt = 0;
  for (int r = 0; r < GRID_ROWS; r++)
    for (int c = mid_col-2; c <= mid_col+2; c++)
      unsafe_cnt += obs[r][c];

  float unsafe_ratio = (float)unsafe_cnt / (float)(GRID_ROWS * 5);

  int left_safe  = 0;
  int right_safe = 0;
  for (int r = 0; r < GRID_ROWS; r++) {
    for (int c = 0; c < mid_col; c++)
      if (!obs[r][c]) left_safe++;
    for (int c = mid_col; c < GRID_COLS; c++)
      if (!obs[r][c]) right_safe++;
  }

  bool turn_left = (left_safe >= right_safe);

  bool obstacle_detected = (unsafe_ratio >= TURN_THRESHOLD);

  pthread_mutex_lock(&pa_mutex);
  plant_avoider_result.obstacle_detected = obstacle_detected;
  plant_avoider_result.safe_col          = mid_col;
  plant_avoider_result.unsafe_ratio      = unsafe_ratio;
  plant_avoider_result.turn_left         = turn_left;
  pthread_mutex_unlock(&pa_mutex);

  return img;
}

void plant_avoider_init(void)
{
  cv_add_to_device(&PLANT_AVOIDER_CAMERA, plant_avoider_cb,
                   PLANT_AVOIDER_FPS, 0);
}