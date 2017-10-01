# CarND-Controls-MPC
Self-Driving Car Engineer Nanodegree Program

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Fortran Compiler
  * Mac: `brew install gcc` (might not be required)
  * Linux: `sudo apt-get install gfortran`. Additionall you have also have to install gcc and g++, `sudo apt-get install gcc g++`. Look in [this Dockerfile](https://github.com/udacity/CarND-MPC-Quizzes/blob/master/Dockerfile) for more info.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * If challenges to installation are encountered (install script fails).  Please review this thread for tips on installing Ipopt.
  * Mac: `brew install ipopt`
       +  Some Mac users have experienced the following error:
       ```
       Listening to port 4567
       Connected!!!
       mpc(4561,0x7ffff1eed3c0) malloc: *** error for object 0x7f911e007600: incorrect checksum for freed object
       - object was probably modified after being freed.
       *** set a breakpoint in malloc_error_break to debug
       ```
       This error has been resolved by updrading ipopt with
       ```brew upgrade ipopt --with-openblas```
       per this [forum post](https://discussions.udacity.com/t/incorrect-checksum-for-freed-object/313433/19).
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/).
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `sudo bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/self-driving-car-sim/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.


## Basic Build Instructions


1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./mpc`.

# Model Predictive Control

The objective of our algorithm will be to optimise the car's actuators (steering, throttle) so that the error between a reference coordinates is minimised.

We are going to follow these steps:
- Calculate the reference center of the lane as a polynomial function
- Calculate the error between the car and the center of the lane for the position of the car and for the steering angle
- Once we know the error that we have we will use a solver that will find the best parameter combination that will minimise this error. The solver will calculate the best solution considering the current parameters and planning for the future up to a specified amount of time
- The solution will give us new values for the steering and for the throttle of the car

## Error calculation

We are given the coordinates of the center of the lane in world's coordinate system as a waypoints. We will first transform those coordinates to car's coordinates and then we will find the third order polynomial that best fit those waypoints:

```
const double cos_psi = cos(psi);
const double sin_psi = sin(psi);

for (i = 0; i < pts_size; i++) {
  const double x_diff = ptsx[i] - px;
  const double y_diff = ptsy[i] - py;
  ptsx_car_coordinate[i] = x_diff * cos_psi + y_diff * sin_psi;  
  ptsy_car_coordinate[i] = -x_diff * sin_psi + y_diff * cos_psi;  
}

Eigen::VectorXd coeffs = polyfit(ptsx_car_coordinate, ptsy_car_coordinate, 3);

```

Once we have the coefficients of the polynomial function, we will then calculate the coordinates of the car's position. As our center coordinates is the center of the car, car's position will always be zero. Then we calculate the coordinate (0, 0) in the polynomial function and any value different to zero will be our error value. Our ideal scenario is to return zero as an error meaning that the car is already in the center of the lane.

In order to calculate the steering angle, we will do it through the derivative of the polynomial function.

```
// we evaluate the desired line against the center of the car -> polyeval(c, px) - py
double cte = polyeval(coeffs, 0); 

// derivative of f(x) = c[1] + 2*c[2]*x + 3*c[3]*x^2 where x is zero
double psi_error = -atan(coeffs[1]); 

```

Once we know our error, we will now solve the optimisation problem knowing:
- The velocity of the car
- The errors cte and psi_error
- The coefficients of the polynomial that fits the center of the lane, as this is our reference

## Optimisation problem solver

The algorithm that we are going to use is an `Interior Point Optimizer` that is used for a large-scale non-linear optimization problems that find local solution. It uses a set of current values, an objective function, a set of constrain functions and upper/lower bounds that constrain the range of the possible parameters.

The optimiser will calculate a number of steps in the future in order to be able to predict better the best possible solution.

The number of seconds in the future that we will predict will be a combination of:
- The number of steps that we want to calculate in the future
- The time between steps

### Time horizon

As we will make a prediction into the future we will have to decide how far in the future we will perform the calculations.

The time horizon will be composed by two factors:
- Number of steps to be computed, the bigger the more number of inputs the optimiser will have to compute adding precision in the calculations. This will add computational overload to the system adding delay in the computations.
- Time difference between steps. At every step we will have an action for an actuator. If the time difference increases will lead to less frequent updates on the actuators.

For a self-driving car the time horizon should be a few seconds at most, as the environment will change too much beyond a few seconds making the predictions invalid.

In our scenario we have choosen `10` steps and the time between steps will be of `0.1`. With these paremeters we will be predicting the best solution up to 1 second in the future. These values give us a good balance between computational cost and actuators update alowing the car to drive safely.

### Current state

Our current state will be use as state variables
- The coordinates of the car
- The current steering angle
- The velocity of the car
- The cte error
- The psi error

### Upper/Lower bounds

- The bounds for the steering angle would be between `[-25, 25]` degrees
- The bounds for the throttle will be `[-1, 1]` as top max and minimum allowed by the simulator
- The rest of the values will have the max/min `1.0e19`

The constrains for these upper and lower bounds will be the current values themselves as those are the values that we have

### Objective function

This function will be calculated across all steps where the step zero is now, and the step + 1 is one step in the future.

The cost to be minimised:
- We will compute the cost accumulated across all the different parameters. This cost can be tuned to minimise or maximise certain parameters
- The cost for the velocity of the car, cte and psi_error will be the standard square error where enhances error with big values and minimises errors with low values
- The cost for the steering will be the squared error but penalised 200 times, the objective is to prevent aggressive steering
- The cost for the throttle will be the squared error but penalised 50 times, the objective is to make the acceleration a bit smoother
- We will also penalise big changes between steps, so that between different time intervals we do not allow sudden change of values.

#### The model

In order to predict the state variables in the future:
```
x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
v_[t+1] = v[t] + a[t] * dt
cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
```

we will use the following formulas where the objective would be to make them equal to zero, as an example:
```
x_[t+1] - (x[t] + v[t] * cos(psi[t]) * dt) = 0 
```

For the cte error `f(x[t])` we will use the coefficients of the polynomial fit at coordinate `x[0]` minutes the coordinate `y[0]` as:
```
const AD<double> first_param = coeffs[0];
const AD<double> second_param = coeffs[1] * x0;
const AD<double> third_param = coeffs[2] * CppAD::pow(x0, 2);
const AD<double> fourth_param = coeffs[3] * CppAD::pow(x0, 3);
AD<double> f0 = first_param + second_param + third_param + fourth_param;

f(x[t]) = f0 - y0
```

For the steering error `psides` we will calculate the arc tangent of the derivative of the polynomial function `f0` as:
```
const AD<double> first_param_derivative = 0;
const AD<double> second_param_derivative = coeffs[1];
const AD<double> third_param_derivative = coeffs[2] * 2 * x0;
const AD<double> fourth_param_derivative = coeffs[3] * 3 * CppAD::pow(x0, 2);
AD<double> f0_derivative = first_param_derivative + second_param_derivative + third_param_derivative + fourth_param_derivative;
AD<double> psides0 = CppAD::atan(f0_derivative);
```

#### Actuators delay

The actuators have a delay of `100ms`. In order to account for it, as we have specify a `dt = 0.1` this will mean that we will skip one step in the constrains. We have accounted for this as:
```
if (t > 0) {
  delta0 = vars[delta_start + t - 1];
  a0 = vars[a_start + t - 1];
}
```