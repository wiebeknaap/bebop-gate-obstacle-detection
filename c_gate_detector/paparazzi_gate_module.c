/*
 * Thin Paparazzi integration template.
 *
 * This file is intentionally conservative: it keeps the detector/guidance core
 * in plain C so it can be compiled standalone, then exposes a small wrapper that
 * can be adapted to the image and guidance hooks in your Paparazzi tree.
 *
 * Replace the include paths and command publication lines to match your setup.
 */

#include "gate_detector_c.h"
#include "gate_guidance_c.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Example placeholders; adapt to your actual Paparazzi includes.
#include "modules/computer_vision/lib/vision/image.h"
#include "state.h"
#include "autopilot.h"
*/

typedef struct {
  gd_workspace_t ws;
  gd_config_t det_cfg;
  bg_config_t guide_cfg;
  bg_state_t guide_state;
  int initialized;
  gd_detection_t last_det;
  bg_command_t last_cmd;
} paparazzi_gate_ctx_t;

static paparazzi_gate_ctx_t g_gate;

int paparazzi_gate_init(int img_w, int img_h) {
  memset(&g_gate, 0, sizeof(g_gate));
  g_gate.det_cfg = gd_default_config();
  g_gate.guide_cfg = bg_default_config();
  bg_reset(&g_gate.guide_state);

  /* These defaults are tuned for runtime more than maximum recall. */
  g_gate.det_cfg.downsample_step = 2;
  g_gate.det_cfg.min_gate_confidence = 0.42f;
  g_gate.det_cfg.open_close_kernel_repeats = 1;

  if (gd_workspace_init(&g_gate.ws, img_w, img_h, g_gate.det_cfg.downsample_step, 512) != 0) {
    return -1;
  }
  g_gate.initialized = 1;
  return 0;
}

void paparazzi_gate_close(void) {
  if (g_gate.initialized) {
    gd_workspace_free(&g_gate.ws);
    g_gate.initialized = 0;
  }
}

int paparazzi_gate_process_bgr(const uint8_t *bgr, int width, int height, int stride_bytes) {
  if (!g_gate.initialized) {
    if (paparazzi_gate_init(width, height) != 0) {
      return -1;
    }
  }

  if (gd_detect_bgr(bgr, width, height, stride_bytes, &g_gate.det_cfg, &g_gate.ws, &g_gate.last_det) < 0) {
    return -2;
  }
  bg_update(&g_gate.guide_cfg, &g_gate.guide_state, &g_gate.last_det, &g_gate.last_cmd);

  /* Adapt this block to your actual command interface.
   * For simulation, the easiest path is often to log these values first:
   *   vx, vy, vz, yaw_rate, confidence, center error.
   * Once that is stable, map them into your velocity or attitude references.
   */
  printf("gate valid=%d conf=%.3f mode=%d ex=%.3f ey=%.3f yaw=%.3f | cmd vx=%.3f vy=%.3f vz=%.3f yr=%.3f\n",
         g_gate.last_det.valid,
         g_gate.last_det.confidence,
         g_gate.last_det.mode,
         g_gate.last_det.lateral_error_norm,
         g_gate.last_det.vertical_error_norm,
         g_gate.last_det.yaw_proxy,
         g_gate.last_cmd.vx,
         g_gate.last_cmd.vy,
         g_gate.last_cmd.vz,
         g_gate.last_cmd.yaw_rate);

  return 0;
}

const gd_detection_t *paparazzi_gate_last_detection(void) { return &g_gate.last_det; }
const bg_command_t *paparazzi_gate_last_command(void) { return &g_gate.last_cmd; }
