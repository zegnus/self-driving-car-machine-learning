#include "NextValsCalculator.h"
#include "utils.hpp"
#include <math.h>

using namespace std;
using namespace utils;

NextValsCalculator::NextValsCalculator() {}

NextValsCalculator::~NextValsCalculator() {}

NextVals NextValsCalculator::Calculate( vector<double> previous_path_x, vector<double> previous_path_y,
                                        double car_x, double car_y, double car_yaw) {
    vector<double> next_x_vals;
    vector<double> next_y_vals;

    double dist_inc = 0.5;

    for (int i = 0; i < 50; i++) {
        next_x_vals.push_back(car_x + (dist_inc * i) * cos(deg2rad(car_yaw)));
        next_y_vals.push_back(car_y + (dist_inc * i) * sin(deg2rad(car_yaw)));
    }

    NextVals nextVals;
    nextVals.next_x_vals_ = next_x_vals;
    nextVals.next_y_vals_ = next_y_vals;

    return nextVals;
}