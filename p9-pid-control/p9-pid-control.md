# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

There's an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`.

## PID Controller

The aim of a PID Controller is to reduce the error between a given value and a reference value. In controlling a car, we can use a PID controller in order to correct the steering (or any other control) so that the position of the car is as close as a reference value as possible.

The PID Controller is composed by three different parameters and an error function.

The error function will give us the error indication against a reference value, being zero the perfect outcome. The objective is to reduce this error tuning the three parameters of the PID controller.

### Proportional parameter
The proportional parameter will increase decrease the error in proportion to the total error. 
- If the proportional parameter is 0.5 and our error is 10, the resulting value is 5
- If the proportional parameter is 0.5 and our error is 2, the resulting value is 1

When the error is zero, then the result will be zero. This causes a problem in solving the steering value of the car because if the car is not perfectly paralel to the reference value (the middle of the lane), when the car goes towards the center of the line and crosses the center, the steering will be modified by the error, being zero, the car will continue crossing the lane. We need then a counter steering parameter.

### Integral parameter
The integral parameter will modify the error accumulated through all the error calculation. This is useful when the initial value has a bias. In the steering example, the steering might have a mechanical issue and be always at an angle different from zero. This parameter will help to reduce the error, when the error accumulates through time (when it shouldn't). Usually a very small value will for this parameter will help to solve the bias.

### Differential parameter
This differential parameter will decrease the error in proportion to the error created between the current step and the previous step. In the steering example, the derivative parameter will help to counter steer when the vehicle is approaching the reference lane.

## Algorithm

The algorithm uses two steps in order to provide the correct steering value:
- We update the current error given a reference value
- The steering value will be directly the negative of the total error

```
pid.UpdateError(cte);
steer_value = - pid.TotalError();
```

When we update the error, we will just calculate the values of:
- Current error, will be the given `cte`
- Differential error between the current error and the previous current error
- Integral error, will be just the sum of all given `cte`

When calculating the total error, we will then multiply all the parameters by all the error and sum them all up:

```=
total_error = Kp_ * p_error_ + Kd_ * d_error_ + Ki_ * i_error_;
```

## Parameter tuning

### SetUp

The initial setup for all threePID parameters is zero. In this scenario the steer value will always be zero regardless of the given error `cte`. We will never steer and fall off the lane in our first turn.

### Parameter tuning

I have manually incremented by 0.1 all three parameters by trial and error using the provided simulator. Implementing `twiddle` with this set-up provided challenging.

The effect of the parameters are the following:
- When the `progressive` parameter is too high the steering is too aggresive when the cte error is big. This results in the car being oscilating heavely around the center of the lane and never converging in it. [Video showing Progressive parameter too big](https://youtu.be/g_G-uj-I91k).
- When the `differential parameter` is too low, the car never manages to settle down in the center of the lane. [Video showing Differential parameter too low](https://youtu.be/gkp1eTPw_Vc)
- The `integral` parameter does not apply in this set-up as the emulator does not have any bias in the steering controls.

### Results

Considering the parameter tuning experiments descrived above and manually incrementing the parameters by steps of `0.1`, the best value for the parameters were:
- Progressive with `0.1`
- Integral with `0`
- Differential with `1.0`

[Video showing final values for PID](https://youtu.be/Gr-nhjVis_E)