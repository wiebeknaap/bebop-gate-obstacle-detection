#ifndef SIMPLE_OBSTACLE_AVOIDER_H
#define SIMPLE_OBSTACLE_AVOIDER_H

#include <stdint.h>
#include <stdbool.h>


struct orange_info {
  float left_fraction;
  float middle_fraction;
  float right_fraction;
  bool left_detected;
  bool middle_detected;
  bool right_detected;
};

enum action {
  FORWARD = 0,
  LEFT,
  RIGHT,
  SEARCH,
  STOP
};

extern float orange_detect_threshold;
extern float middle_strong_threshold;
extern int low_conf_threshold;
extern int high_conf_threshold;
extern int max_confidence;

extern struct orange_info orange_raw;
extern struct orange_info orange_filtered;
extern struct command last_command;
extern enum action last_action;
extern int obstacle_confidence;

void simple_obstacle_avoider_init(void);
void simple_obstacle_avoider_periodic(void);

void set_orange_fractions(float left_fraction, float middle_fraction, float right_fraction);

void update_detection_flags(struct orange_info *orange);
int update_confidence(const struct orange_info *orange);
enum action decide_action(const struct orange_info *orange, int confidence);
const char *action_name(enum action action);

#endif