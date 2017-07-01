# Extended Kalman Filter Project

The source code of this project can be found in [github](https://github.com/zegnus/self-driving-car-machine-learning/tree/master/p6-extended-kalman-filters)

The objective of this project is to build an Extended Kalman Filter that will estimate a moving object with noisy lidar and radar measurements.

We are provided by a data file that contains lidar and radar measurements in the directory `/data`.

We can also run the predictions with the provided [simulator](https://github.com/udacity/self-driving-car-sim/releases) that will connect through [Websocket](https://github.com/uWebSockets/uWebSockets) to our program, will provide the measurements to our program and we will provide the results back to the simulator.

# Projet set-up

The project is written in c++ and has the following dependencies for linux:
- cmake >= 3.5
- make >= 4.1
- gcc/g++ >= 5.4

There is also provided an installation script for mac and linux at `./install-ubuntu.sh` and `/.install-mac.sh` that will install websocket and other dependencies. Be aware of the websocket dependencies if you have conda with websocket installed in it, as the soft links might cause a conflict.

# How to execute the project

## Command line

Follow this commands once you have the simulator running:
```
cd build
cmake .. 
make
./ExtendedKF
```

## Using an IDE

We can set-up easely [Eclipse](https://eclipse.org/cdt/) (and others) following the instructions under `./ide_profiles/Eclipse/` and [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html)

# Algorithm description and steps

The algorithm will calculate and update the matrices involved in the Kalman and Extended Kalman Filter, initialise the values on first measurement, prevent divisions by zero and provide a Root Mean Square Error of our results.

The matrices involved are the following:
**P**: transition state uncertainty covariance matrix
**F**: transition state matrix
**Q**: process prediction uncertainty covariance matrix
**H**: measurement matrix that will remove the velocity from the measurement for Lidar
**R**: measurement uncertainty covariance matrix, provided by the sensor manufacturer

## Telemetry input
We will get a telemetry measurement that will contain data from Lidar or Radar, and also the timestamp of the measurement
    * For Lidar we will get the position (x and y)
    * For Radar we will get polar coordinates (range, bearing and radial velocity)

## Initialisation
We will then initialise the position and velocity from the measurement, the transition matrix, the covariance and the timestamp. In case of the radar we will have to convert the polar coordinates to cartesian in order to use the same calculations on both measurements:
```
VectorXd FusionEKF::initialisePositionVelocity(const MeasurementPackage &measurement_pack) {
  VectorXd x = VectorXd(4);

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

  if (fabs(position_x) < 0.0001) position_x = 0.0001;
  if (fabs(position_y) < 0.0001) position_y = 0.0001;

  x << position_x, position_y, velocity_x, velocity_y;
  return x;
}
```

## Prediction
On the following measurement we will update the transition matrix `F` and the process transition noise matrix `Q` with the new timestamp and a provided noise value of 9.0f and then will make a prediction updating the position matrix `x = F * x` and the transition matrix `P = F * P * F_trans + Q`

```
MatrixXd FusionEKF::createProcessNoiseMatrixQwith(float elapsedTime, float noise_ax, float noise_ay) {
  float elapsedTime_power_2 = elapsedTime * elapsedTime;
  float elapsedTime_power_3 = elapsedTime_power_2 * elapsedTime;
  float elapsedTime_power_4 = elapsedTime_power_3 * elapsedTime;

  float elapsedTime_power_4_divided_by_4 = elapsedTime_power_4 / 4;
  float elapsedtime_power_3_divided_by_2 = elapsedTime_power_3 / 2;

  MatrixXd Q = MatrixXd(4, 4);
  Q << elapsedTime_power_4_divided_by_4 * noise_ax, 0, elapsedtime_power_3_divided_by_2 * noise_ax, 0,
        0, elapsedTime_power_4_divided_by_4 * noise_ay, 0, elapsedtime_power_3_divided_by_2 * noise_ay,
        elapsedtime_power_3_divided_by_2 * noise_ax, 0, elapsedTime_power_2 * noise_ax, 0,
        0, elapsedtime_power_3_divided_by_2 * noise_ay, 0, elapsedTime_power_2 * noise_ay;
  return Q;
}
```

## Update
We will then make an update considering the measurement error covariance for Lidar or Radar, and also calculating the measurement matrix `H` through the `Jacobian` for the Radar measurement, and the `Identity` for x and y for the Lidar measurement.
 * Lidar update will use `y = z - (H * x)`
 * Radar update will calculate `y = z - H(x)_jacobian` by which `H(x) = range, angle and radial velocity` calculated from the cartesian input. We will also normalise the angle component of `y` to be between `-pi, +pi`

Finally we will update the position matrix:
```
S = H * P * Ht + R)
K = P * H_trans * S_inverse
x = x + (K * y)
```

And the transition matrix covariance:
```
P = (Identity_matrix - (K * H)) * P
```
