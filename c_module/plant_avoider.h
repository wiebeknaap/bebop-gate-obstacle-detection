#ifndef PLANT_AVOIDER_H
#define PLANT_AVOIDER_H

#include "std.h"

#define GRID_ROWS 25
#define GRID_COLS 20

#ifndef PLANT_AVOIDER_CAMERA
#define PLANT_AVOIDER_CAMERA front_camera
#endif
#ifndef PLANT_AVOIDER_FPS
#define PLANT_AVOIDER_FPS 4
#endif

#define SOBEL_THRESHOLD        30
#define EDGE_DENSITY_THRESHOLD 0.10f
#define MIN_VERTICAL_FILL      0.03f
#define TURN_THRESHOLD         0.4f

#define ROI_X_START 0.1f
#define ROI_X_END   0.9f
#define ROI_Y_START 0.1f
#define ROI_Y_END   0.9f

struct plant_avoider_result_t {
  bool     obstacle_detected;
  int32_t  safe_col;
  float    unsafe_ratio;
  bool     turn_left;
};

extern struct plant_avoider_result_t plant_avoider_result;

extern void plant_avoider_init(void);

#endif