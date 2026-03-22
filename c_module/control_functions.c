float oa_color_count_frac = 1.0f;
float moveDistance = 0.5f;
float oob_heading_increment = 25.f;
const int16_t max_trajectory_confidence = 2;
float oa_heading_increment = 25.f;
static bool obstacle_entry = false;

case SAFE:
  obstacle_entry = true;
  obstacle_free_confidence = max_trajectory_confidence;
  moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
  if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
    navigation_state = OUT_OF_BOUNDS;
  } else if (obstacle_free_confidence == 0
             || plant_avoider_result.obstacle_detected) {
    navigation_state = OBSTACLE_FOUND;
  } else {
    float col_offset = (plant_avoider_result.safe_col - GRID_COLS / 2.f) / (GRID_COLS / 2.f);
    increase_nav_heading(col_offset * oa_heading_increment);
    moveWaypointForward(WP_GOAL, moveDistance);
  }
  break;

case OBSTACLE_FOUND:
  if (obstacle_entry) {
    waypoint_move_here_2d(WP_GOAL);
    waypoint_move_here_2d(WP_TRAJECTORY);
    obstacle_entry = false;
  }
  if (plant_avoider_result.obstacle_detected) {
    float increment = plant_avoider_result.turn_left
                      ? -oa_heading_increment
                      :  oa_heading_increment;
    increase_nav_heading(increment);
  } else {
    navigation_state = SAFE;
  }
  break;

case OUT_OF_BOUNDS:
  waypoint_move_here_2d(WP_GOAL);
  waypoint_move_here_2d(WP_TRAJECTORY);
  increase_nav_heading(plant_avoider_result.turn_left
                       ? -oob_heading_increment
                       :  oob_heading_increment);
  moveWaypointForward(WP_TRAJECTORY, 1.5f);
  if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
    navigation_state = SAFE;
  }
  break;