#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

class FusionEKF {
public:
  /**
   * Constructor.
   */
  FusionEKF();

  /**
   * Destructor.
   */
  virtual ~FusionEKF();

  /**
   * Run the whole flow of the Kalman Filter from here.
   */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
   * Kalman Filter update and prediction math lives in here.
   */
  KalmanFilter ekf_;

private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp in microseconds
  long long previous_timestamp_;

  // tool object used to compute Jacobian and RMSE
  Tools tools;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd Hj_;

  void initialisePositionVelocity(const MeasurementPackage &measurement_pack);
  void initialiseTransitionStateMatrixPWithCovariance(int covariance);
  void updateLocalTimestamp(const MeasurementPackage &measurement_pack);
  float calculateElapsedTime(const MeasurementPackage &measurement_pack);
  void updateTransitionMatrixPWithElapsedTime(float elapsedTime);
  void updateProcessNoiseMatrixQwith(float elapsedTime, int noise_ax, int noise_ay);
};

#endif /* FusionEKF_H_ */
