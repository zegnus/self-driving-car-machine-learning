#ifndef NEXT_VALS_H
#define NEXT_VALS_H

#include <vector>

class NextVals {
    public:
        std::vector<double> next_x_vals_;
        std::vector<double> next_y_vals_;

        NextVals();
        virtual ~NextVals();   
};

#endif