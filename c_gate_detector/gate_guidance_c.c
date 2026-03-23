#include "gate_guidance_c.h"

#include <math.h>
#include <string.h>

static float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

bg_config_t bg_default_config(void) {
  bg_config_t cfg;
  cfg.k_lat = 0.9f;
  cfg.k_vert = 0.8f;
  cfg.k_yaw = 1.0f;
  cfg.search_yaw_rate = 0.25f;
  cfg.approach_vx = 0.35f;
  cfg.commit_vx = 0.65f;
  cfg.pass_vx = 0.80f;
  cfg.align_tol = 0.12f;
  cfg.commit_tol = 0.05f;
  cfg.confidence_acquire = 0.45f;
  cfg.lost_frame_limit = 10;
  return cfg;
}

void bg_reset(bg_state_t *st) {
  memset(st, 0, sizeof(*st));
  st->mode = BG_SEARCH;
}

void bg_update(const bg_config_t *cfg_in, bg_state_t *st, const gd_detection_t *det, bg_command_t *cmd) {
  bg_config_t cfg = cfg_in ? *cfg_in : bg_default_config();
  memset(cmd, 0, sizeof(*cmd));

  const int has_gate = det && det->valid && det->confidence >= cfg.confidence_acquire;
  if (has_gate) {
    st->good_frames++;
    st->lost_frames = 0;
  } else {
    st->lost_frames++;
    st->good_frames = 0;
  }

  if (st->lost_frames > cfg.lost_frame_limit) {
    st->mode = BG_LOST;
  } else if (st->mode == BG_SEARCH && has_gate) {
    st->mode = BG_ACQUIRE;
  }

  if (!has_gate) {
    if (st->mode == BG_LOST || st->mode == BG_SEARCH) {
      st->mode = BG_SEARCH;
      cmd->yaw_rate = cfg.search_yaw_rate;
      cmd->mode = st->mode;
      return;
    }
    cmd->yaw_rate = 0.15f;
    cmd->mode = st->mode;
    return;
  }

  const float ex = det->lateral_error_norm;
  const float ey = det->vertical_error_norm;
  const float ealign = fmaxf(fabsf(ex), fabsf(ey));

  if (st->mode == BG_ACQUIRE && st->good_frames >= 2) st->mode = BG_ALIGN;
  if (st->mode == BG_ALIGN && ealign < cfg.align_tol) st->mode = BG_APPROACH;
  if (st->mode == BG_APPROACH && ealign < cfg.commit_tol && det->opening_w > 0.18f * (float)det->image_width) st->mode = BG_COMMIT;
  if (st->mode == BG_COMMIT && det->opening_w > 0.35f * (float)det->image_width) st->mode = BG_PASS;
  if (st->mode == BG_PASS && det->opening_w < 0.10f * (float)det->image_width) st->mode = BG_SEARCH;

  cmd->vy = clampf(-cfg.k_lat * ex, -0.6f, 0.6f);
  cmd->vz = clampf(-cfg.k_vert * ey, -0.5f, 0.5f);
  cmd->yaw_rate = clampf(-cfg.k_yaw * det->yaw_proxy - 0.4f * ex, -0.8f, 0.8f);

  switch (st->mode) {
    case BG_SEARCH:
      cmd->vx = 0.0f;
      cmd->yaw_rate = cfg.search_yaw_rate;
      break;
    case BG_ACQUIRE:
      cmd->vx = 0.0f;
      break;
    case BG_ALIGN:
      cmd->vx = 0.05f;
      break;
    case BG_APPROACH:
      cmd->vx = cfg.approach_vx;
      break;
    case BG_COMMIT:
      cmd->vx = cfg.commit_vx;
      cmd->vy *= 0.5f;
      cmd->vz *= 0.5f;
      break;
    case BG_PASS:
      cmd->vx = cfg.pass_vx;
      cmd->vy = 0.0f;
      cmd->vz = 0.0f;
      cmd->yaw_rate = 0.0f;
      break;
    case BG_LOST:
    default:
      cmd->vx = 0.0f;
      cmd->vy = 0.0f;
      cmd->vz = 0.0f;
      cmd->yaw_rate = cfg.search_yaw_rate;
      st->mode = BG_SEARCH;
      break;
  }

  cmd->mode = st->mode;
}
