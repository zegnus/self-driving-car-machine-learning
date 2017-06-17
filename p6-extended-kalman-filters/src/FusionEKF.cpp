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

    // Initialise position and velocity

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

    cout << "EKF init: " << ekf_.x_ << endl;

    // Initialise transition state with hight uncertainty covariance matrix P

    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1000, 0, 0, 0,
                0, 1000, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;


    // Initialise timestamp

    previous_timestamp_ = measurement_pack.timestamp_;

    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
   * Update the state transition matrix F according to the new elapsed time.
   - Time is measured in seconds.
   * Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

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
