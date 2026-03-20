#ifndef PLANT_AVOIDER_H
#define PLANT_AVOIDER_H

#include <stdint.h>
#include "modules/computer_vision/lib/vision/image.h"

#define GRID_ROWS              8
#define GRID_COLS              10
#define SOBEL_THRESHOLD        60
#define EDGE_DENSITY_THRESHOLD 0.25f
#define MIN_VERTICAL_FILL      0.03f

#define ROI_X_START  0.10f
#define ROI_X_END    0.90f
#define ROI_Y_START  0.10f
#define ROI_Y_END    0.90f

typedef struct {
    int   safe_col;
    int   obstacle_detected;
    float cell_density[GRID_ROWS][GRID_COLS];
    int   cell_obstacle[GRID_ROWS][GRID_COLS];
} PlantAvoiderResult;

void plant_avoider_init(void);
void plant_avoider_periodic(void);
struct image_t *plant_avoider_detect(struct image_t *img, uint8_t camera_id);

extern PlantAvoiderResult plant_avoider_result;

#endif /* PLANT_AVOIDER_H */