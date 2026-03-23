#ifndef GATE_DETECTOR_C_H
#define GATE_DETECTOR_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GD_MODE_NONE = 0,
  GD_MODE_POST_PAIR = 1
} gd_mode_t;

typedef struct {
  int downsample_step;
  int rotate_portrait;
  int min_component_area;
  float min_vertical_aspect;
  float min_vertical_height_ratio;
  float min_gate_confidence;
  int min_bg_diff;
  int min_br_diff;
  int min_saturation;
  int min_blue_value;
  int min_gap_px;
  int max_gap_px;
  int open_close_kernel_repeats;
} gd_config_t;

typedef struct {
  int valid;
  float confidence;
  gd_mode_t mode;
  int image_width;
  int image_height;
  float center_x;
  float center_y;
  int outer_x;
  int outer_y;
  int outer_w;
  int outer_h;
  int opening_x;
  int opening_y;
  int opening_w;
  int opening_h;
  float lateral_error_norm;
  float vertical_error_norm;
  float yaw_proxy;
  int proc_width;
  int proc_height;
  int num_components;
} gd_detection_t;

typedef struct {
  int x;
  int y;
  int w;
  int h;
  int area;
  float cx;
  float cy;
  float aspect;
} gd_component_t;

typedef struct {
  int src_width;
  int src_height;
  int proc_width;
  int proc_height;
  int ds_width;
  int ds_height;
  int max_components;
  uint8_t *mask;
  uint8_t *tmp;
  uint8_t *visited;
  int *queue;
  uint32_t *integral;
  gd_component_t *components;
} gd_workspace_t;

gd_config_t gd_default_config(void);
int gd_workspace_init(gd_workspace_t *ws, int src_width, int src_height, int downsample_step, int max_components);
void gd_workspace_free(gd_workspace_t *ws);

int gd_detect_bgr(
    const uint8_t *bgr,
    int width,
    int height,
    int stride_bytes,
    const gd_config_t *cfg,
    gd_workspace_t *ws,
    gd_detection_t *out);

#ifdef __cplusplus
}
#endif

#endif
