#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
* Initializes Unscented Kalman filter
*/
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:
  Complete the initialization. See ukf.h for other member properties.
  Hint: one or more values initialized above might be wildly off...
  */

  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // time when the state is true, in us
  time_us_ = 0.0;

  // state dimension
  n_x_ = 5;

  n_radar_ = 3;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //create vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  R_radar_ = MatrixXd(n_radar_, n_radar_);

  R_radar_ << std_radr_*std_radr_, 0, 0,
                         0, std_radphi_*std_radphi_, 0,
                         0, 0, std_radrd_*std_radrd_;

  // the current NIS for radar
  NIS_radar_ = 0.0;

  // the current NIS for laser
  NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
* @param {MeasurementPackage} meas_package The latest measurement data of
* either radar or laser.
*/
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */



    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {
      /**
      TODO:
      * Initialize the state x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
      */
      /**
      Initialize state.
      */

      // init covariance matrix
      P_ << 1,    0, 0, 0, 0,
               0, 1, 0, 0, 0,
               0,    0, 1, 0, 0,
               0,    0, 0, 1, 0,
               0,    0, 0, 0, 1;

      // init timestamp
      time_us_ = meas_package.timestamp_;

      float position_x = 1;
      float position_y = 1;
      float velocity_absolute = 1;
      float yaw_angle = 1;
      float yaw_rate = 1;

      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        /**
         Convert radar from polar to cartesian coordinates and initialize state.
         */
        float horizontal_projection = cos(meas_package.raw_measurements_[1]);
        float vertical_projection = sin(meas_package.raw_measurements_[1]);

        position_x = meas_package.raw_measurements_[0] * horizontal_projection;
        position_y = meas_package.raw_measurements_[0] * vertical_projection;
        float velocity_x = meas_package.raw_measurements_[2] * horizontal_projection;
        float velocity_y = meas_package.raw_measurements_[2] * vertical_projection;
        velocity_absolute = sqrt(velocity_x * velocity_x + velocity_y * velocity_y);
      } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        position_x = meas_package.raw_measurements_[0];
        position_y = meas_package.raw_measurements_[1];
      }

      if (fabs(position_x) < 0.0001) position_x = 0.0001;
      if (fabs(position_y) < 0.0001) position_y = 0.0001;

      x_ << position_x, position_y, velocity_absolute, yaw_angle, yaw_rate;

      // done initializing, no need to predict or update
      is_initialized_ = true;

      return;
    }

    /*****************************************************************************
    *  Prediction
    ****************************************************************************/
    //compute the time elapsed between the current and previous measurements
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;  //dt - expressed in seconds
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    /*****************************************************************************
    *  Update
    ****************************************************************************/

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      //UpdateLidar(meas_package);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    }
}

/**
* Predicts sigma points, the state, and the state covariance matrix.
* @param {double} delta_t the change in time (in seconds) between the last
* measurement and this one.
*/
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  /*****************************************************************************
  *  Augment Sigma Points
  ****************************************************************************/
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //set lambda for augmented sigma points

  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_*std_a_;
  P_aug(6, 6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    MatrixXd value = sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1) = x_aug + value;
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - value;
  }

  /*****************************************************************************
  *  Predict Sigma Points
  ****************************************************************************/
  //predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //extract values for better readability
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  /*****************************************************************************
  *  Convert Predicted Sigma Points to Mean/Covariance
  ****************************************************************************/

  //predicted state mean
  x_.fill(0.0);             //******* necessary? *********
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);             //******* necessary? *********
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    normaliseAngleIn(&(x_diff(3)));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

}

void UKF::normaliseAngleIn(double *angle) {
  while (*angle < - M_PI) {
    *angle += 2 * M_PI;
  }

  while (*angle > M_PI) {
    *angle -= 2 * M_PI;
  }
}

/**
* Updates the state and the state covariance matrix using a laser measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateLidar(MeasurementPackage meas_package) {
//  /**
//  TODO:
//  Complete this function! Use lidar data to update the belief about the object's
//  position. Modify the state vector, x_, and covariance, P_.
//  You'll also need to calculate the lidar NIS.
//  */
//
//  //extract measurement as VectorXd
//  VectorXd z = meas_package.raw_measurements_;
//
//  //set measurement dimension, lidar can measure p_x and p_y
//  int n_z = 2;
//
//  //create matrix for sigma points in measurement space
//  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
//
//  //transform sigma points into measurement space
//  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
//
//    // extract values for better readibility
//    double p_x = Xsig_pred_(0, i);
//    double p_y = Xsig_pred_(1, i);
//
//    // measurement model
//    Zsig(0, i) = p_x;
//    Zsig(1, i) = p_y;
//  }
//
//  //mean predicted measurement
//  VectorXd z_pred = VectorXd(n_z);
//  z_pred.fill(0.0);
//  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
//    z_pred = z_pred + weights_(i) * Zsig.col(i);
//  }
//
//  //measurement covariance matrix S
//  MatrixXd S = MatrixXd(n_z, n_z);
//  S.fill(0.0);
//  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
//
//    //residual
//    VectorXd z_diff = Zsig.col(i) - z_pred;
//
//    S = S + weights_(i) * z_diff * z_diff.transpose();
//  }
//
//  //add measurement noise covariance matrix
//  MatrixXd R = MatrixXd(n_z, n_z);
//  R << std_laspx_*std_laspx_, 0,
//       0, std_laspy_*std_laspy_;
//  S = S + R;
//
//  //create matrix for cross correlation Tc
//  MatrixXd Tc = MatrixXd(n_x_, n_z);
//
//  /*****************************************************************************
//  *  UKF Update for Lidar
//  ****************************************************************************/
//  //calculate cross correlation matrix
//  Tc.fill(0.0);
//  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
//
//    //residual
//    VectorXd z_diff = Zsig.col(i) - z_pred;
//
//    // state difference
//    VectorXd x_diff = Xsig_pred_.col(i) - x_;
//
//    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
//  }
//
//  //Kalman gain K;
//  MatrixXd K = Tc * S.inverse();
//
//  //residual
//  VectorXd z_diff = z - z_pred;
//
//  //calculate NIS
//  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
//
//  //update state mean and covariance matrix
//  x_ = x_ + K * z_diff;
//  P_ = P_ - K*S*K.transpose();

}

/**
* Updates the state and the state covariance matrix using a radar measurement.
* @param {MeasurementPackage} meas_package
*/
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //extract measurement as VectorXd
  VectorXd z = VectorXd(n_radar_);
  z <<  meas_package.raw_measurements_(0),
        meas_package.raw_measurements_(1),
        meas_package.raw_measurements_(2);


  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_radar_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1, i) = atan2(p_y, p_x);                                 //phi
    Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_radar_);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_radar_, n_radar_);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    normaliseAngleIn(&(z_diff(1)));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix

  S = S + R_radar_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_radar_);

  /*****************************************************************************
  *  UKF Update for Radar
  ****************************************************************************/
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    normaliseAngleIn(&(z_diff(1)));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    normaliseAngleIn(&(x_diff(3)));

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  normaliseAngleIn(&(z_diff(1)));

  //calculate NIS
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

}
