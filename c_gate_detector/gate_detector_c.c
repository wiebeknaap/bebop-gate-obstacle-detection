#include "gate_detector_c.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static inline int max_i(int a, int b) { return a > b ? a : b; }
static inline int min_i(int a, int b) { return a < b ? a : b; }
static inline float max_f(float a, float b) { return a > b ? a : b; }
static inline float min_f(float a, float b) { return a < b ? a : b; }

static void zero_detection(gd_detection_t *out, int width, int height) {
  memset(out, 0, sizeof(*out));
  out->image_width = width;
  out->image_height = height;
  out->mode = GD_MODE_NONE;
}

gd_config_t gd_default_config(void) {
  gd_config_t cfg;
  cfg.downsample_step = 2;
  cfg.rotate_portrait = 1;
  cfg.min_component_area = 180;
  cfg.min_vertical_aspect = 1.2f;
  cfg.min_vertical_height_ratio = 0.08f;
  cfg.min_gate_confidence = 0.42f;
  cfg.min_bg_diff = 24;
  cfg.min_br_diff = 18;
  cfg.min_saturation = 42;
  cfg.min_blue_value = 50;
  cfg.min_gap_px = 18;
  cfg.max_gap_px = 100000;
  cfg.open_close_kernel_repeats = 1;
  return cfg;
}

int gd_workspace_init(gd_workspace_t *ws, int src_width, int src_height, int downsample_step, int max_components) {
  if (!ws || src_width <= 0 || src_height <= 0 || downsample_step <= 0 || max_components <= 0) {
    return -1;
  }

  memset(ws, 0, sizeof(*ws));
  ws->src_width = src_width;
  ws->src_height = src_height;
  if (src_width >= src_height) {
    ws->proc_width = src_width;
    ws->proc_height = src_height;
  } else {
    ws->proc_width = src_height;
    ws->proc_height = src_width;
  }
  ws->ds_width = (ws->proc_width + downsample_step - 1) / downsample_step;
  ws->ds_height = (ws->proc_height + downsample_step - 1) / downsample_step;
  ws->max_components = max_components;

  const size_t n = (size_t)ws->ds_width * (size_t)ws->ds_height;
  ws->mask = (uint8_t *)calloc(n, sizeof(uint8_t));
  ws->tmp = (uint8_t *)calloc(n, sizeof(uint8_t));
  ws->visited = (uint8_t *)calloc(n, sizeof(uint8_t));
  ws->queue = (int *)malloc(n * sizeof(int));
  ws->integral = (uint32_t *)calloc((size_t)(ws->ds_width + 1) * (size_t)(ws->ds_height + 1), sizeof(uint32_t));
  ws->components = (gd_component_t *)calloc((size_t)max_components, sizeof(gd_component_t));

  if (!ws->mask || !ws->tmp || !ws->visited || !ws->queue || !ws->integral || !ws->components) {
    gd_workspace_free(ws);
    return -2;
  }
  return 0;
}

void gd_workspace_free(gd_workspace_t *ws) {
  if (!ws) return;
  free(ws->mask);
  free(ws->tmp);
  free(ws->visited);
  free(ws->queue);
  free(ws->integral);
  free(ws->components);
  memset(ws, 0, sizeof(*ws));
}

static void erode3x3(const uint8_t *src, uint8_t *dst, int w, int h) {
  memset(dst, 0, (size_t)w * (size_t)h);
  for (int y = 1; y < h - 1; y++) {
    for (int x = 1; x < w - 1; x++) {
      int ok = 1;
      for (int ky = -1; ky <= 1 && ok; ky++) {
        const uint8_t *row = src + (size_t)(y + ky) * (size_t)w;
        for (int kx = -1; kx <= 1; kx++) {
          if (!row[x + kx]) {
            ok = 0;
            break;
          }
        }
      }
      dst[(size_t)y * (size_t)w + (size_t)x] = ok ? 1u : 0u;
    }
  }
}

static void dilate3x3(const uint8_t *src, uint8_t *dst, int w, int h) {
  memset(dst, 0, (size_t)w * (size_t)h);
  for (int y = 1; y < h - 1; y++) {
    for (int x = 1; x < w - 1; x++) {
      int ok = 0;
      for (int ky = -1; ky <= 1 && !ok; ky++) {
        const uint8_t *row = src + (size_t)(y + ky) * (size_t)w;
        for (int kx = -1; kx <= 1; kx++) {
          if (row[x + kx]) {
            ok = 1;
            break;
          }
        }
      }
      dst[(size_t)y * (size_t)w + (size_t)x] = ok ? 1u : 0u;
    }
  }
}

static void morphology_open_close(uint8_t *mask, uint8_t *tmp, int w, int h, int repeats) {
  for (int i = 0; i < repeats; i++) {
    erode3x3(mask, tmp, w, h);
    dilate3x3(tmp, mask, w, h);
    dilate3x3(mask, tmp, w, h);
    erode3x3(tmp, mask, w, h);
  }
}

static void build_blue_mask_downsampled(
    const uint8_t *bgr,
    int width,
    int height,
    int stride,
    const gd_config_t *cfg,
    gd_workspace_t *ws) {
  const int step = cfg->downsample_step;
  const int dw = ws->ds_width;
  const int dh = ws->ds_height;
  const int rotate = cfg->rotate_portrait && (height > width);

  for (int y = 0; y < dh; y++) {
    for (int x = 0; x < dw; x++) {
      const int xp = min_i(x * step, ws->proc_width - 1);
      const int yp = min_i(y * step, ws->proc_height - 1);
      int xs, ys;
      if (rotate) {
        xs = width - 1 - yp;
        ys = xp;
      } else {
        xs = xp;
        ys = yp;
      }
      xs = clampi(xs, 0, width - 1);
      ys = clampi(ys, 0, height - 1);
      const uint8_t *p = bgr + (size_t)ys * (size_t)stride + 3 * xs;
      const int b = p[0];
      const int g = p[1];
      const int r = p[2];
      const int maxc = max_i(b, max_i(g, r));
      const int minc = min_i(b, min_i(g, r));
      const int sat = maxc > 0 ? (255 * (maxc - minc)) / maxc : 0;
      const int is_blue = (b >= cfg->min_blue_value) &&
                          (b - g > cfg->min_bg_diff) &&
                          (b - r > cfg->min_br_diff) &&
                          (sat >= cfg->min_saturation);
      ws->mask[(size_t)y * (size_t)dw + (size_t)x] = is_blue ? 1u : 0u;
    }
  }
}

static int extract_components(
    gd_workspace_t *ws,
    const gd_config_t *cfg,
    gd_component_t *verticals,
    int *num_verticals,
    gd_component_t *horizontals,
    int *num_horizontals) {

  const int w = ws->ds_width;
  const int h = ws->ds_height;
  const int min_area = max_i(4, cfg->min_component_area / max_i(1, cfg->downsample_step * cfg->downsample_step));
  int ncomp = 0;
  *num_verticals = 0;
  *num_horizontals = 0;
  memset(ws->visited, 0, (size_t)w * (size_t)h);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      const size_t idx0 = (size_t)y * (size_t)w + (size_t)x;
      if (!ws->mask[idx0] || ws->visited[idx0]) continue;

      int qh = 0;
      int qt = 0;
      ws->queue[qt++] = (int)idx0;
      ws->visited[idx0] = 1u;

      int area = 0;
      int minx = x, maxx = x, miny = y, maxy = y;
      int64_t sumx = 0;
      int64_t sumy = 0;

      while (qh < qt) {
        int idx = ws->queue[qh++];
        int cx = idx % w;
        int cy = idx / w;
        area++;
        sumx += cx;
        sumy += cy;
        if (cx < minx) minx = cx;
        if (cx > maxx) maxx = cx;
        if (cy < miny) miny = cy;
        if (cy > maxy) maxy = cy;

        for (int ky = -1; ky <= 1; ky++) {
          int ny = cy + ky;
          if (ny < 0 || ny >= h) continue;
          for (int kx = -1; kx <= 1; kx++) {
            int nx = cx + kx;
            if (nx < 0 || nx >= w) continue;
            size_t nidx = (size_t)ny * (size_t)w + (size_t)nx;
            if (ws->mask[nidx] && !ws->visited[nidx]) {
              ws->visited[nidx] = 1u;
              ws->queue[qt++] = (int)nidx;
            }
          }
        }
      }

      if (area < min_area) continue;
      if (ncomp >= ws->max_components) continue;

      gd_component_t c;
      c.x = minx;
      c.y = miny;
      c.w = maxx - minx + 1;
      c.h = maxy - miny + 1;
      c.area = area;
      c.cx = (float)sumx / (float)area;
      c.cy = (float)sumy / (float)area;
      c.aspect = (float)c.h / (float)max_i(c.w, 1);
      ws->components[ncomp++] = c;

      if (c.aspect >= cfg->min_vertical_aspect && c.h >= cfg->min_vertical_height_ratio * (float)h) {
        verticals[*num_verticals] = c;
        (*num_verticals)++;
      }
      if (((float)c.w / (float)max_i(c.h, 1)) >= 1.35f && c.w >= 0.08f * (float)w) {
        horizontals[*num_horizontals] = c;
        (*num_horizontals)++;
      }
    }
  }

  return ncomp;
}

static int cmp_cx(const void *a, const void *b) {
  const gd_component_t *ca = (const gd_component_t *)a;
  const gd_component_t *cb = (const gd_component_t *)b;
  if (ca->cx < cb->cx) return -1;
  if (ca->cx > cb->cx) return 1;
  return 0;
}

static void build_integral_image(gd_workspace_t *ws) {
  const int w = ws->ds_width;
  const int h = ws->ds_height;
  const int iw = w + 1;
  memset(ws->integral, 0, (size_t)iw * (size_t)(h + 1) * sizeof(uint32_t));

  for (int y = 0; y < h; y++) {
    uint32_t row_sum = 0;
    for (int x = 0; x < w; x++) {
      row_sum += ws->mask[(size_t)y * (size_t)w + (size_t)x] ? 1u : 0u;
      ws->integral[(size_t)(y + 1) * (size_t)iw + (size_t)(x + 1)] =
          ws->integral[(size_t)y * (size_t)iw + (size_t)(x + 1)] + row_sum;
    }
  }
}

static uint32_t rect_sum(const gd_workspace_t *ws, int x1, int y1, int x2, int y2) {
  const int w = ws->ds_width;
  const int h = ws->ds_height;
  const int iw = w + 1;
  x1 = clampi(x1, 0, w);
  x2 = clampi(x2, 0, w);
  y1 = clampi(y1, 0, h);
  y2 = clampi(y2, 0, h);
  if (x2 <= x1 || y2 <= y1) return 0;
  const uint32_t *ii = ws->integral;
  return ii[(size_t)y2 * (size_t)iw + (size_t)x2]
       - ii[(size_t)y1 * (size_t)iw + (size_t)x2]
       - ii[(size_t)y2 * (size_t)iw + (size_t)x1]
       + ii[(size_t)y1 * (size_t)iw + (size_t)x1];
}

static float rect_occupancy(const gd_workspace_t *ws, int x1, int y1, int x2, int y2) {
  x1 = clampi(x1, 0, ws->ds_width);
  x2 = clampi(x2, 0, ws->ds_width);
  y1 = clampi(y1, 0, ws->ds_height);
  y2 = clampi(y2, 0, ws->ds_height);
  const int area = max_i(1, (x2 - x1) * (y2 - y1));
  return (float)rect_sum(ws, x1, y1, x2, y2) / (float)area;
}

static void scale_detection_to_fullres(gd_detection_t *out, int step) {
  out->outer_x *= step;
  out->outer_y *= step;
  out->outer_w *= step;
  out->outer_h *= step;
  out->opening_x *= step;
  out->opening_y *= step;
  out->opening_w *= step;
  out->opening_h *= step;
  out->center_x *= (float)step;
  out->center_y *= (float)step;
}

int gd_detect_bgr(
    const uint8_t *bgr,
    int width,
    int height,
    int stride_bytes,
    const gd_config_t *cfg_in,
    gd_workspace_t *ws,
    gd_detection_t *out) {

  if (!bgr || !ws || !out || width <= 0 || height <= 0 || stride_bytes < 3 * width) {
    return -1;
  }

  gd_config_t cfg = cfg_in ? *cfg_in : gd_default_config();
  zero_detection(out, width, height);
  out->proc_width = ws->proc_width;
  out->proc_height = ws->proc_height;

  const int expected_proc_w = (cfg.rotate_portrait && height > width) ? height : width;
  const int expected_proc_h = (cfg.rotate_portrait && height > width) ? width : height;
  if (ws->src_width != width || ws->src_height != height || ws->proc_width != expected_proc_w || ws->proc_height != expected_proc_h || ws->ds_width != (expected_proc_w + cfg.downsample_step - 1) / cfg.downsample_step || ws->ds_height != (expected_proc_h + cfg.downsample_step - 1) / cfg.downsample_step) {
    return -2;
  }

  build_blue_mask_downsampled(bgr, width, height, stride_bytes, &cfg, ws);
  morphology_open_close(ws->mask, ws->tmp, ws->ds_width, ws->ds_height, max_i(1, cfg.open_close_kernel_repeats));
  build_integral_image(ws);

  gd_component_t *verticals = (gd_component_t *)malloc((size_t)ws->max_components * sizeof(gd_component_t));
  gd_component_t *horizontals = (gd_component_t *)malloc((size_t)ws->max_components * sizeof(gd_component_t));
  if (!verticals || !horizontals) {
    free(verticals);
    free(horizontals);
    return -3;
  }

  int num_verticals = 0;
  int num_horizontals = 0;
  out->num_components = extract_components(ws, &cfg, verticals, &num_verticals, horizontals, &num_horizontals);
  if (num_verticals > 1) {
    qsort(verticals, (size_t)num_verticals, sizeof(gd_component_t), cmp_cx);
  }

  float best_score = -1e9f;
  gd_detection_t best = *out;
  best.valid = 0;
  best.mode = GD_MODE_NONE;

  for (int i = 0; i < num_verticals; i++) {
    for (int j = i + 1; j < num_verticals; j++) {
      const gd_component_t left = verticals[i];
      const gd_component_t right = verticals[j];
      const int gap = right.x - (left.x + left.w);
      if (gap < cfg.min_gap_px / max_i(cfg.downsample_step, 1)) continue;
      if (gap > min_i((int)(0.78f * (float)ws->ds_width), cfg.max_gap_px / max_i(cfg.downsample_step, 1))) continue;

      const int overlap = max_i(0, min_i(left.y + left.h, right.y + right.h) - max_i(left.y, right.y));
      const float overlap_ratio = (float)overlap / (float)max_i(min_i(left.h, right.h), 1);
      if (overlap_ratio < 0.40f) continue;

      const float height_ratio = (float)min_i(left.h, right.h) / (float)max_i(left.h, right.h);
      const float width_ratio = (float)min_i(left.w, right.w) / (float)max_i(left.w, right.w);
      const float avg_height = 0.5f * (float)(left.h + right.h);
      const float gap_ratio = (float)gap / max_f(avg_height, 1.0f);
      const float gap_score = max_f(0.0f, 1.0f - fabsf(gap_ratio - 0.72f) / 0.85f);

      const int top_y = min_i(left.y, right.y);
      const int bottom_y = max_i(left.y + left.h, right.y + right.h);
      const int x_outer_1 = left.x;
      const int x_outer_2 = right.x + right.w;
      const int x_inner_1 = left.x + left.w;
      const int x_inner_2 = right.x;
      if (x_inner_2 <= x_inner_1) continue;

      const int band = max_i(2, (int)(0.06f * max_i(left.h, right.h)));
      const float top_occ = rect_occupancy(ws, x_outer_1, top_y - band, x_outer_2, top_y + band + 1);
      const float opening_occ = rect_occupancy(ws, x_inner_1, top_y + band, x_inner_2, bottom_y - band);
      const float hollow_score = max_f(0.0f, 1.0f - opening_occ / 0.18f);

      float horizontal_support = 0.0f;
      for (int k = 0; k < num_horizontals; k++) {
        const gd_component_t hcomp = horizontals[k];
        const int spans_pair = (hcomp.x <= x_inner_1) && ((hcomp.x + hcomp.w) >= x_inner_2);
        const int near_top = abs(hcomp.y - top_y) <= (int)(0.20f * max_i(left.h, right.h));
        if (spans_pair && near_top) {
          const float support = min_f(1.0f, (float)hcomp.w / (float)max_i(x_outer_2 - x_outer_1, 1));
          if (support > horizontal_support) horizontal_support = support;
        }
      }

      const float symmetry_score = 0.6f * height_ratio + 0.4f * width_ratio;
      const float top_support_score = min_f(1.0f, 0.65f * top_occ + 0.35f * horizontal_support);
      float edge_penalty = 0.0f;
      if (left.x < 2 || right.x + right.w > ws->ds_width - 2) edge_penalty += 0.12f;
      if ((x_outer_2 - x_outer_1) < 0.12f * (float)ws->ds_width) edge_penalty += 0.10f;

      const float score = 0.24f * symmetry_score +
                          0.22f * overlap_ratio +
                          0.22f * gap_score +
                          0.20f * top_support_score +
                          0.12f * hollow_score - edge_penalty;

      if (score > best_score) {
        best_score = score;
        best.valid = score >= cfg.min_gate_confidence ? 1 : 0;
        best.confidence = score;
        best.mode = GD_MODE_POST_PAIR;
        best.outer_x = x_outer_1;
        best.outer_y = top_y;
        best.outer_w = x_outer_2 - x_outer_1;
        best.outer_h = bottom_y - top_y;
        best.opening_x = x_inner_1;
        best.opening_y = top_y;
        best.opening_w = x_inner_2 - x_inner_1;
        best.opening_h = bottom_y - top_y;
        best.center_x = (float)x_inner_1 + 0.5f * (float)(x_inner_2 - x_inner_1);
        best.center_y = (float)top_y + 0.5f * (float)(bottom_y - top_y);
        const float width_asym = (float)(right.w - left.w) / (float)max_i(right.w + left.w, 1);
        const float height_asym = (float)(right.h - left.h) / (float)max_i(right.h + left.h, 1);
        best.yaw_proxy = 0.65f * width_asym + 0.35f * height_asym;
      }
    }
  }

  free(verticals);
  free(horizontals);

  if (best_score <= -1e8f) {
    return 0;
  }

  scale_detection_to_fullres(&best, cfg.downsample_step);
  best.lateral_error_norm = (best.center_x - 0.5f * (float)ws->proc_width) / max_f(0.5f * (float)ws->proc_width, 1.0f);
  best.vertical_error_norm = (best.center_y - 0.5f * (float)ws->proc_height) / max_f(0.5f * (float)ws->proc_height, 1.0f);
  *out = best;
  out->image_width = width;
  out->image_height = height;
  out->proc_width = ws->proc_width;
  out->proc_height = ws->proc_height;
  return 1;
}
