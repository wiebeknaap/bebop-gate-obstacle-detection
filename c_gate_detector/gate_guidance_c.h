#ifndef GATE_GUIDANCE_C_H
#define GATE_GUIDANCE_C_H

#include "gate_detector_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  BG_SEARCH = 0,
  BG_ACQUIRE = 1,
  BG_ALIGN = 2,
  BG_APPROACH = 3,
  BG_COMMIT = 4,
  BG_PASS = 5,
  BG_LOST = 6
} bg_mode_t;

typedef struct {
  float vx;
  float vy;
  float vz;
  float yaw_rate;
  bg_mode_t mode;
} bg_command_t;

typedef struct {
  bg_mode_t mode;
  int lost_frames;
  int good_frames;
} bg_state_t;

typedef struct {
  float k_lat;
  float k_vert;
  float k_yaw;
  float search_yaw_rate;
  float approach_vx;
  float commit_vx;
  float pass_vx;
  float align_tol;
  float commit_tol;
  float confidence_acquire;
  int lost_frame_limit;
} bg_config_t;

bg_config_t bg_default_config(void);
void bg_reset(bg_state_t *st);
void bg_update(const bg_config_t *cfg, bg_state_t *st, const gd_detection_t *det, bg_command_t *cmd);

#ifdef __cplusplus
}
#endif

#endif
