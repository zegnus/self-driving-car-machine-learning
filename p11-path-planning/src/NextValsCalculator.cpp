#include "NextValsCalculator.h"
#include "utils.hpp"
#include "spline.h"
#include <cmath>
#include <iostream>

using namespace std;
using namespace utils;

const double maximum_allowed_velocity = 49.5;
const double minimum_allowed_speed_increment = 0.224;
const double maximum_allowed_speed_increment = minimum_allowed_speed_increment * 4;
const double safety_distance_front_car = 20;
const double safety_distance_front_car_other_lane = safety_distance_front_car + 12;
const double safety_distance_back_car_other_lane = 8;
const double safety_distance_following_car = 10;
const double slow_down_parameter = (safety_distance_front_car - safety_distance_following_car) / maximum_allowed_speed_increment;

NextValsCalculator::NextValsCalculator() {}

NextValsCalculator::~NextValsCalculator() {}

bool same_lane_car(double other_car_d, double car_lane) {
    return other_car_d < car_lane + 2 && other_car_d > car_lane - 2;
}

bool left_lane_car(double other_car_d, double car_lane) {
    return other_car_d < car_lane - 2 && other_car_d > car_lane - 6;
}

bool right_lane_car(double other_car_d, double car_lane) {
    return other_car_d > car_lane + 2 && other_car_d < car_lane + 6;
}

bool different_lane_car_is_too_close(double other_car_s, double car_s) {
    double car_distance = abs(other_car_s - car_s);

    if (other_car_s > car_s && car_distance < safety_distance_front_car_other_lane) {
        return true;
    }

    if (other_car_s < car_s && car_distance < safety_distance_back_car_other_lane) {
        return true;
    }

    return false;
}

bool same_lane_car_is_too_close(double other_car_s, double car_s) {
    return other_car_s > car_s && other_car_s - car_s < safety_distance_front_car;
}

bool changing_lanes(double curent_d, double target_d) {
    return abs(curent_d - target_d) > 1;
}

double slow_down(double car_separation) {
    double distance_to_safety = car_separation - safety_distance_following_car;
    if (distance_to_safety < 0) {
        // emergency break
        return maximum_allowed_speed_increment;
    } else {
        return maximum_allowed_speed_increment - (distance_to_safety / slow_down_parameter);
    }
}

NextVals NextValsCalculator::Calculate( vector<double> previous_path_x, vector<double> previous_path_y,
                                        double car_x, double car_y, double car_s, double car_d, double car_yaw,
                                        vector<double> map_waypoints_x, vector<double> map_waypoints_y, vector<double> map_waypoints_s,
                                        int &lane, double &ref_vel,
                                        vector<vector<double>> sensor_fusion,
                                        double end_path_s, double end_path_d) {
    const int prev_size = previous_path_x.size();
    const double car_lane = 2 + 4 * lane;

    if (prev_size > 0) {
        car_s = end_path_s;
    }

    bool stay_in_lane_at_maximum_speed = true;
    bool follow_car_in_front = false;
    bool safe_to_move_left_lane = true;
    bool safe_to_move_right_lane = true;

    double target_car_in_front_speed;
    double target_car_in_front_position = safety_distance_front_car;

    const int sensor_fusion_size = sensor_fusion.size();
    for (int i = 0; i < sensor_fusion_size; i++) {
        float other_car_d = sensor_fusion[i][6];

        double other_car_vx = sensor_fusion[i][3];
        double other_car_vy = sensor_fusion[i][4];
        double other_car_speed = sqrt(other_car_vx*other_car_vx + other_car_vy*other_car_vy); 
        double other_car_s = sensor_fusion[i][5];
        other_car_s += (double)prev_size * 0.02 * other_car_speed; // this is the future position of the other car

        other_car_speed *= 2.24;

        if (same_lane_car(other_car_d, car_lane)) {
            if (same_lane_car_is_too_close(other_car_s, car_s)) {
                // cout << "same lane car: " << other_car_s - car_s;
                stay_in_lane_at_maximum_speed = false;
                follow_car_in_front = true;

                target_car_in_front_speed = other_car_speed;
                target_car_in_front_position = other_car_s;
            } 
        } else if (left_lane_car(other_car_d, car_lane)) {
            if (different_lane_car_is_too_close(other_car_s, car_s)) {
                safe_to_move_left_lane = false;
                target_car_in_front_speed = other_car_speed;
            }
        } else if (right_lane_car(other_car_d, car_lane)) {
            if (different_lane_car_is_too_close(other_car_s, car_s)) {
                safe_to_move_right_lane = false;
                target_car_in_front_speed = other_car_speed;
            }
        }
    }

    if (changing_lanes(car_d, car_lane)) {
        // do-nothing
    } else {
        if (stay_in_lane_at_maximum_speed) {
            if (ref_vel < maximum_allowed_velocity) {
                ref_vel += maximum_allowed_speed_increment;
            }
        } else if (lane > 0 && safe_to_move_left_lane) {
            lane--;
            if (ref_vel > target_car_in_front_speed) {
                ref_vel -= maximum_allowed_speed_increment;
            }
        } else if (lane < 2 && safe_to_move_right_lane) {
            lane++;
            if (ref_vel > target_car_in_front_speed) {
                ref_vel -= maximum_allowed_speed_increment;
            }
        } else if (follow_car_in_front) {
            const double car_separation = target_car_in_front_position - car_s;
            const double car_speed_decrement = slow_down(car_separation);
            cout << " ref_vel: " << ref_vel << ", target_vel: " << target_car_in_front_speed << ", car_speed_decrement: " << car_speed_decrement << endl;
            if (ref_vel > target_car_in_front_speed) {
                ref_vel -= car_speed_decrement;
            }
        }

        if (ref_vel > maximum_allowed_velocity) {
            ref_vel = maximum_allowed_velocity;
        }
    }

    vector<double> anchor_points_x;
    vector<double> anchor_points_y;

    double ref_x = car_x;
    double ref_y = car_y;
    double ref_yaw = deg2rad(car_yaw);

    // WE CREATE 5 POINTS

    if (prev_size < 2) {
        const double prev_car_x = car_x - cos(car_yaw);
        const double prev_car_y = car_y - sin(car_yaw);

        anchor_points_x.push_back(prev_car_x);
        anchor_points_x.push_back(car_x);

        anchor_points_y.push_back(prev_car_y);
        anchor_points_y.push_back(car_y);
    } else {
        ref_x = previous_path_x[prev_size - 1];
        ref_y = previous_path_y[prev_size - 1];

        const double ref_x_prev = previous_path_x[prev_size - 2];
        const double ref_y_prev = previous_path_y[prev_size - 2];
        ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

        anchor_points_x.push_back(ref_x_prev);
        anchor_points_x.push_back(ref_x);

        anchor_points_y.push_back(ref_y_prev);
        anchor_points_y.push_back(ref_y);
    }

    vector<double> next_wp0 = getXY(car_s + 30, car_lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
    vector<double> next_wp1 = getXY(car_s + 60, car_lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);
    vector<double> next_wp2 = getXY(car_s + 90, car_lane, map_waypoints_s, map_waypoints_x, map_waypoints_y);

    anchor_points_x.push_back(next_wp0[0]);
    anchor_points_x.push_back(next_wp1[0]);
    anchor_points_x.push_back(next_wp2[0]);

    anchor_points_y.push_back(next_wp0[1]);
    anchor_points_y.push_back(next_wp1[1]);
    anchor_points_y.push_back(next_wp2[1]);

    // WE ROTATE THE COORDINATES SO THAT THE YAW OF THE CAR IS ZERO

    const int anchor_points_x_size = anchor_points_x.size();
    for (int i = 0; i < anchor_points_x_size; i++) {
        const double shift_x = anchor_points_x[i] - ref_x;
        const double shift_y = anchor_points_y[i] - ref_y;

        anchor_points_x[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
        anchor_points_y[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
    }

    tk::spline spline;
    spline.set_points(anchor_points_x, anchor_points_y);

    vector<double> future_path_x;
    vector<double> future_path_y;

    for (int i = 0; i < previous_path_x.size(); i++) {
        future_path_x.push_back(previous_path_x[i]);
        future_path_y.push_back(previous_path_y[i]);
    }

    // N_POINTS x 0.02_SECONDS x VELOCITY_DESIRED m/s = TARGET_DISTANCE
    const double target_x = target_car_in_front_position;
    const double target_y = spline(target_x);
    const double target_dist = sqrt((target_x * target_x) + (target_y * target_y));    
    const double N = target_dist / (0.02 * ref_vel / 2.24); // 2.24 MPH -> meters per second
    const double distance_between_points = target_x / N;

    double x_add_on = 0;

    for (int i = 1; i <= 50 - previous_path_x.size(); i++) {
        double x_point = x_add_on + distance_between_points;
        double y_point = spline(x_point);

        x_add_on = x_point;

        double x_ref = x_point;
        double y_ref = y_point;

        // WE ROTATE BACK TO NORMAL
        x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
        y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

        x_point += ref_x;
        y_point += ref_y;

        future_path_x.push_back(x_point);
        future_path_y.push_back(y_point);
    }

    NextVals nextVals;
    nextVals.next_x_vals_ = future_path_x;
    nextVals.next_y_vals_ = future_path_y;

    return nextVals;
}