#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  Kp_ = Kp;
  Ki_ = Ki;
  Kd_ = Kd;

  p_error_ = 0;
  i_error_ = 0;
  d_error_ = 0;
}

void PID::UpdateError(double cte) {
  p_error_ = cte;

  d_error_ = cte - prev_cte_;
  prev_cte_ = cte;

  i_error_ += cte;
}

double PID::TotalError() {
  return Kp_ * p_error_
      + Kd_ * d_error_
      + Ki_ * i_error_;
}

