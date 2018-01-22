#ifndef NEXT_VALS_CALCULATOR_H
#define NEXT_VALS_CALCULATOR_H

#include "NextVals.h"

class NextValsCalculator {
    public:
        NextValsCalculator();
        virtual ~NextValsCalculator();
        NextVals Calculate( std::vector<double> previous_path_x, std::vector<double> previous_path_y,
                            double car_x, double car_y, double car_yaw);
};

#endif