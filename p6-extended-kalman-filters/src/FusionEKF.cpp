#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement matrix that will remove the velocity from the measurement from Lidar
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    cout << "EKF: " << endl;

    initialisePositionVelocity(measurement_pack);
    initialiseTransitionStateMatrixPWithCovariance(1000);
    updateLocalTimestamp(measurement_pack);

    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float elapsedTime = calculateElapsedTime(measurement_pack);
  updateLocalTimestamp(measurement_pack);

  updateTransitionMatrixPWithElapsedTime(elapsedTime);

  int noise_ax = 9;
  int noise_ay = 9;
  updateProcessNoiseMatrixQwith(elapsedTime, noise_ax, noise_ay);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
   * Use the sensor type to perform the update step.
   * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
  } else {
    // Laser updates
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

void FusionEKF::initialisePositionVelocity(const MeasurementPackage &measurement_pack) {
  ekf_.x_ = VectorXd(4);

  float position_x = 1;
  float position_y = 1;
  float velocity_x = 1;
  float velocity_y = 1;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    /**
     Convert radar from polar to cartesian coordinates and initialize state.
     */
    float horizontal_projection = cos(measurement_pack.raw_measurements_[1]);
    float vertical_projection = sin(measurement_pack.raw_measurements_[1]);

    position_x = measurement_pack.raw_measurements_[0] * horizontal_projection;
    position_y = measurement_pack.raw_measurements_[0] * vertical_projection;
    velocity_x = measurement_pack.raw_measurements_[2] * horizontal_projection;
    velocity_y = measurement_pack.raw_measurements_[2] * vertical_projection;
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    position_x = measurement_pack.raw_measurements_[0];
    position_y = measurement_pack.raw_measurements_[1];
  }

  if (position_x == 0) position_x = 0.0001;
  if (position_y == 0) position_y = 0.0001;

  ekf_.x_ << position_x, position_y, velocity_x, velocity_y;
}

void FusionEKF::initialiseTransitionStateMatrixPWithCovariance(int covariance) {
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << covariance, 0, 0, 0,
              0, covariance, 0, 0,
              0, 0, covariance, 0,
              0, 0, 0, covariance;
}

void FusionEKF::updateLocalTimestamp(const MeasurementPackage &measurement_pack) {
  previous_timestamp_ = measurement_pack.timestamp_;
}

float FusionEKF::calculateElapsedTime(const MeasurementPackage &measurement_pack) {
  float elapsedTime = measurement_pack.timestamp_ - previous_timestamp_;
  return elapsedTime / 1000000.0; // in seconds
}

void FusionEKF::updateTransitionMatrixPWithElapsedTime(float elapsedTime) {
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, elapsedTime, 0,
              0, 1, 0, elapsedTime,
              0, 0, 1, 0,
              0, 0, 0, 1;
}

void FusionEKF::updateProcessNoiseMatrixQwith(float elapsedTime, int noise_ax, int noise_ay) {
  float elapsedTime_power_2 = elapsedTime * elapsedTime;
  float elapsedTime_power_3 = elapsedTime_power_2 * elapsedTime;
  float elapsedTime_power_4 = elapsedTime_power_3 * elapsedTime;

  float elapsedTime_power_4_divided_by_4 = elapsedTime_power_4 / 4;
  float elapsedtime_power_3_divided_by_2 = elapsedTime_power_3 / 2;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << elapsedTime_power_4_divided_by_4 * noise_ax, 0, elapsedtime_power_3_divided_by_2 * noise_ax, 0,
            0, elapsedTime_power_4_divided_by_4 * noise_ay, 0, elapsedtime_power_3_divided_by_2 * noise_ay,
            elapsedtime_power_3_divided_by_2 * noise_ax, 0, elapsedTime_power_2 * noise_ax, 0,
            0, elapsedtime_power_3_divided_by_2 * noise_ay, 0, elapsedTime_power_2 * noise_ay;
}
