#ifndef NEXT_VALS_CALCULATOR_H
#define NEXT_VALS_CALCULATOR_H

#include "NextVals.h"

class NextValsCalculator {
    public:
        NextValsCalculator();
        virtual ~NextValsCalculator();
        NextVals Calculate( std::vector<double> previous_path_x, std::vector<double> previous_path_y,
                            double car_x, double car_y, double car_s, double car_d, double car_yaw,
                            std::vector<double> map_waypoints_x, std::vector<double> map_waypoints_y, std::vector<double> map_waypoints_s,
                            int &lane, double &ref_vel,
                            std::vector<std::vector<double>> sensor_fusion,
                            double end_path_s, double end_path_d);
};

#endif