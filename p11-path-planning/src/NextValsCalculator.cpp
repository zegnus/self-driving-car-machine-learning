#include "NextValsCalculator.h"
#include "utils.hpp"
#include <math.h>
#include "spline.h"

using namespace std;
using namespace utils;

NextValsCalculator::NextValsCalculator() {}

NextValsCalculator::~NextValsCalculator() {}

NextVals NextValsCalculator::Calculate( vector<double> previous_path_x, vector<double> previous_path_y,
                                        double car_x, double car_y, double car_s, double car_yaw,
                                        vector<double> map_waypoints_x, vector<double> map_waypoints_y, vector<double> map_waypoints_s,
                                        int lane, double ref_vel) {
    int prev_size = previous_path_x.size();

    vector<double> anchor_points_x;
    vector<double> anchor_points_y;

    double ref_x = car_x;
    double ref_y = car_y;
    double ref_yaw = deg2rad(car_yaw);

    // WE CREATE 5 POINTS

    if (prev_size < 2) {
        double prev_car_x = car_x - cos(car_yaw);
        double prev_car_y = car_y - sin(car_yaw);

        anchor_points_x.push_back(prev_car_x);
        anchor_points_x.push_back(car_x);

        anchor_points_y.push_back(prev_car_y);
        anchor_points_y.push_back(car_y);
    } else {
        ref_x = previous_path_x[prev_size - 1];
        ref_y = previous_path_y[prev_size - 1];

        double ref_x_prev = previous_path_x[prev_size - 2];
        double ref_y_prev = previous_path_y[prev_size - 2];
        ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

        anchor_points_x.push_back(ref_x_prev);
        anchor_points_x.push_back(ref_x);

        anchor_points_y.push_back(ref_y_prev);
        anchor_points_y.push_back(ref_y);
    }

    double car_lane = 2 + 4 * lane;

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

    for (int i = 0; i < anchor_points_x.size(); i++) {
        double shift_x = anchor_points_x[i] - ref_x;
        double shift_y = anchor_points_y[i] - ref_y;

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
    double target_x = 30.0;
    double target_y = spline(target_x);
    double target_dist = sqrt((target_x * target_x) + (target_y * target_y));    
    double N = target_dist / (0.02 * ref_vel / 2.24); // 2.24 MPH -> meters per second
    double distance_between_points = target_x / N;

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

    // double dist_inc = 0.5;

    // for (int i = 0; i < 50; i++) {
    //     double next_s = car_s + (i + 1) * dist_inc;
    //     double next_d = 6;
    //     vector<double> xy = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
    //     future_path_x.push_back(xy[0]);
    //     future_path_y.push_back(xy[1]);
    // }

    NextVals nextVals;
    nextVals.next_x_vals_ = future_path_x;
    nextVals.next_y_vals_ = future_path_y;

    return nextVals;
}