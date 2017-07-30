#include "ukf.h"
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
  std_a_ = 2; // 3 - 30

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3; //0.7 - 30

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

  is_initialized_ = false;

  n_x_ = 5; // state vector dimension

  n_radar_ = 3; // radar state vector dimension

  n_lidar_ = 2; // radar state vector dimension

  n_aug_ = 7; // augmented state vector dimension

  lambda_ = 3 - n_aug_;

  x_sigma_points_predicted_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  previous_timestamp_ = 0;

  NIS_radar_ = 0;

  NIS_lidar_ = 0;

  R_radar_ = MatrixXd(n_radar_, n_radar_);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(n_lidar_, n_lidar_);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;

  // initialise weights

  weights_ = VectorXd(2 * n_aug_ + 1);

  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) { //2n+1 weights
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  cout << "ProcessMeasurement" << endl;

  // initialise if necessary
  if (!is_initialized_) {
    cout << "Initialise Unscented Kalman Filter" << endl;

    x_ = InitialiseStateVector(meas_package);
    P_ = InitialiseCovarianceMatrix(1);

    updateLocalTimestamp(meas_package);
    is_initialized_ = true;

    return;
  }

  float delta_t = calculateElapsedTime(meas_package);
  updateLocalTimestamp(meas_package);

  cout << "delta_t" << endl;
  cout << delta_t << endl;

  cout << "Prediction" << endl;

  // prediction
  Prediction(delta_t);

  // update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "UpdateRadar" << endl;
    UpdateRadar(meas_package);
  } else {
    cout << "UpdateLidar" << endl;
    UpdateLidar(meas_package);
  }

  cout << "end" << endl;
}

void UKF::updateLocalTimestamp(const MeasurementPackage meas_package) {
  previous_timestamp_ = meas_package.timestamp_;
}

float UKF::calculateElapsedTime(const MeasurementPackage meas_package) {
  long elapsedTime = meas_package.timestamp_ - previous_timestamp_;
  return elapsedTime / 1000000.0; // in seconds
}

VectorXd UKF::InitialiseStateVector(const MeasurementPackage meas_package) {
  VectorXd x = VectorXd(5);

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

  x << position_x, position_y, velocity_absolute, yaw_angle, yaw_rate;
  return x;
}

MatrixXd UKF::InitialiseCovarianceMatrix(float covariance) {
  MatrixXd P = MatrixXd(5, 5);

  P <<  covariance, 0, 0, 0, 0,
        0, covariance, 0, 0, 0,
        0, 0, covariance, 0, 0,
        0, 0, 0, covariance, 0,
        0, 0, 0, 0, covariance;

  return P;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(float delta_t) {

  cout << "start prediction" << endl;

  VectorXd x_augmented = GeneratedAugmentedState();
  MatrixXd P_augmented = GenerateAugmentedCovarianceMatrix();
  MatrixXd x_sigma_points_augmented = GenerateAugmentedSigmaPoints(x_augmented,
                                                                   P_augmented);

  PredictSigmaPoints(x_sigma_points_augmented, delta_t);

  PredictStateAndCovariance();

  // x_ = PredictStateVector(x_sigma_points_predicted_);

  cout << "x_: " << x_ << endl ;

  // P_ = PredictCovarianceMatrix(x_, x_sigma_points_predicted_);

  cout << "P_: " << P_ << endl;

  cout << "end prediction" << endl;

}

VectorXd UKF::GeneratedAugmentedState() {
  VectorXd x_augmented = VectorXd(n_aug_);

  x_augmented.fill(0.0);
  x_augmented.head(n_x_) = x_;

  return x_augmented;
}

MatrixXd UKF::GenerateAugmentedCovarianceMatrix() {
  MatrixXd P_augmented = MatrixXd(n_aug_, n_aug_);

  P_augmented.fill(0.0);
  P_augmented.topLeftCorner(5, 5) = P_;
  P_augmented(5, 5) = std_a_ * std_a_;
  P_augmented(6, 6) = std_yawdd_ * std_yawdd_;

  return P_augmented;
}

MatrixXd UKF::GenerateAugmentedSigmaPoints(const VectorXd x_augmented,
                                           const MatrixXd P_augmented) {
  MatrixXd x_sigma_points_augmented = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  MatrixXd L = P_augmented.llt().matrixL(); // square root matrix

  x_sigma_points_augmented.col(0) = x_augmented;

  for (int i = 0; i < n_aug_; i++) {
    MatrixXd value = sqrt(lambda_ + n_aug_) * L.col(i);
    x_sigma_points_augmented.col(i + 1) = x_augmented + value;
    x_sigma_points_augmented.col(i + 1 + n_aug_) = x_augmented - value;
  }

  return x_sigma_points_augmented;
}

void UKF::PredictSigmaPoints(const MatrixXd x_sigma_points_augmented,
                             const float delta_t) {
  // [0] -> position x
  // [1] -> position y
  // [2] -> velocity
  // [3] -> yaw angle [yaw]
  // [4] -> yaw rate
  // [5] -> longitudinal acceleration noise [nu_acceleration]
  // [6] -> yaw acceleration noise [nu_yaw_acceleration]

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //extract values for better readability
    double p_x = x_sigma_points_augmented(0, i);
    double p_y = x_sigma_points_augmented(1, i);
    double v = x_sigma_points_augmented(2, i);
    double yaw = x_sigma_points_augmented(3, i);
    double yawd = x_sigma_points_augmented(4, i);
    double nu_a = x_sigma_points_augmented(5, i);
    double nu_yawdd = x_sigma_points_augmented(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    //write predicted sigma point into right column
    x_sigma_points_predicted_(0, i) = px_p;
    x_sigma_points_predicted_(1, i) = py_p;
    x_sigma_points_predicted_(2, i) = v_p;
    x_sigma_points_predicted_(3, i) = yaw_p;
    x_sigma_points_predicted_(4, i) = yawd_p;
  }
}

void UKF::PredictStateAndCovariance() {
  x_.fill(0.0);
  P_.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) { //iterate over sigma points
    x_ = x_ + weights_(i) * x_sigma_points_predicted_.col(i);

    VectorXd x_diff = x_sigma_points_predicted_.col(i) - x_;
    normaliseAngleIn(&(x_diff(3)));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::normaliseAngleIn(double *angle) {
  while (*angle < -M_PI) {
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
  MatrixXd z_sigma_points = TransformSigmaPointsToLidarSpace();

  VectorXd z_predicted = VectorXd(n_lidar_);
  z_predicted = PredictMean(z_sigma_points, z_predicted);

  MatrixXd S = PredictLidarCovarianceMatrix(z_sigma_points, z_predicted);
  S = S + R_lidar_;

  MatrixXd T = CalculateLidarCrossCorrelationMatrix(z_sigma_points,
                                                    z_predicted);

  MatrixXd K = T * S.inverse(); // Kalman Gain

  VectorXd z = VectorXd(n_lidar_);
  z <<  meas_package.raw_measurements_(0),
        meas_package.raw_measurements_(1);

  //residual
  VectorXd z_diff = z - z_predicted;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  MatrixXd z_sigma_points = TransformSigmaPointsToRadarSpace();

  VectorXd z_predicted = VectorXd(n_radar_);
  z_predicted = PredictMean(z_sigma_points, z_predicted);

  MatrixXd S = PredictRadarCovarianceMatrix(z_sigma_points, z_predicted);
  S = S + R_radar_;

  MatrixXd T = CalculateRadarCrossCorrelationMatrix(z_sigma_points,
                                                    z_predicted);
  MatrixXd K = T * S.inverse(); // Kalman Gain

  VectorXd z = VectorXd(n_radar_);
  z <<  meas_package.raw_measurements_(0),
        meas_package.raw_measurements_(1),
        meas_package.raw_measurements_(2);

  //residual
  VectorXd z_diff = z - z_predicted;
  normaliseAngleIn(&(z_diff(1)));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

MatrixXd UKF::TransformSigmaPointsToRadarSpace() {
  MatrixXd z_sigma_points = MatrixXd(n_radar_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points

    // extract values for better readibility
    double p_x = x_sigma_points_predicted_(0, i);
    double p_y = x_sigma_points_predicted_(1, i);
    double v = x_sigma_points_predicted_(2, i);
    double yaw = x_sigma_points_predicted_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    z_sigma_points(0, i) = sqrt(p_x * p_x + p_y * p_y); //r
    z_sigma_points(1, i) = atan2(p_y, p_x); //phi
    z_sigma_points(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
  }

  return z_sigma_points;
}

MatrixXd UKF::TransformSigmaPointsToLidarSpace() {
  MatrixXd z_sigma_points = MatrixXd(n_lidar_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points

    // extract values for better readibility
    double p_x = x_sigma_points_predicted_(0, i);
    double p_y = x_sigma_points_predicted_(1, i);

    // measurement model
    z_sigma_points(0, i) = p_x;
    z_sigma_points(1, i) = p_y;
  }

  return z_sigma_points;
}

VectorXd UKF::PredictMean(const MatrixXd z_sigma_points, VectorXd &z_predicted) {
  z_predicted.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_predicted = z_predicted + weights_(i) * z_sigma_points.col(i);
  }

  return z_predicted;
}

MatrixXd UKF::PredictRadarCovarianceMatrix(const MatrixXd z_sigma_points,
                                           const MatrixXd z_predicted) {
  MatrixXd S = MatrixXd(n_radar_, n_radar_);
  S.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points
    //residual
    VectorXd z_diff = z_sigma_points.col(i) - z_predicted;
    normaliseAngleIn(&(z_diff(1)));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  return S;
}

MatrixXd UKF::PredictLidarCovarianceMatrix(const MatrixXd z_sigma_points,
                                           const MatrixXd z_predicted) {
  MatrixXd S = MatrixXd(n_lidar_, n_lidar_);
  S.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points
    //residual
    VectorXd z_diff = z_sigma_points.col(i) - z_predicted;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  return S;
}

MatrixXd UKF::CalculateRadarCrossCorrelationMatrix(
                                                   const MatrixXd z_sigma_points,
                                                   const MatrixXd z_predicted) {
  MatrixXd T = MatrixXd(n_x_, n_radar_);

  T.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points
    //residual
    VectorXd z_diff = z_sigma_points.col(i) - z_predicted;
    normaliseAngleIn(&(z_diff(1)));

    // state difference
    VectorXd x_diff = x_sigma_points_predicted_.col(i) - x_;
    normaliseAngleIn(&(x_diff(3)));

    T = T + weights_(i) * x_diff * z_diff.transpose();
  }

  return T;
}

MatrixXd UKF::CalculateLidarCrossCorrelationMatrix(
                                                   const MatrixXd z_sigma_points,
                                                   const MatrixXd z_predicted) {
  MatrixXd T = MatrixXd(n_x_, n_lidar_);

  T.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) { //2n+1 simga points
    VectorXd z_diff = z_sigma_points.col(i) - z_predicted;
    VectorXd x_diff = x_sigma_points_predicted_.col(i) - x_;
    T = T + weights_(i) * x_diff * z_diff.transpose();
  }

  return T;
}
