#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  //x_ = (F_ * x_) + u;
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // We update for the standard Kalman filter (Lidar input)
  VectorXd y = z - (H_ * x_);
  updateStateAndTransitionState(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // We update for the Extended Kalman Filter (Radar input)
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  VectorXd h_x = VectorXd(3);
  float range = sqrt(px*px + py*py);
  float angle = atan(py / px);
  float radial_velocity = (px*vx + py*vy) / range;
  h_x << range, angle, radial_velocity;

  VectorXd y = z - h_x;
  updateStateAndTransitionState(y);
}

void KalmanFilter::updateStateAndTransitionState(const VectorXd &y) {
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();

  x_ = x_ + (K * y);

  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - (K * H_)) * P_;
}
