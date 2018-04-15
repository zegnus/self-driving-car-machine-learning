#include "yaw_controller.h"

using namespace DBWNODE_NS;

using namespace std;

YawController::YawController()
{}

YawController::~YawController()
{}

double YawController::get_angle(double radius)
{
    double angle = atan(this->wheel_base_ / radius) * this->steer_ratio_;
    return max(min_angle_, min(max_angle_, angle));
}

double YawController::get_steering(double linear_velocity, double angular_velocity, double current_velocity)
{
    double steering_wheel_angle = 0.0;
    if(fabs(current_velocity) > 0.5)
    {
        steering_wheel_angle = steer_ratio_ * atan(wheel_base_ * angular_velocity / current_velocity);
    }
    else
    {
        if(fabs(linear_velocity) > 0.1)
        {
            steering_wheel_angle = steer_ratio_ * atan(wheel_base_ * angular_velocity / linear_velocity);
        }
        else
        {
            steering_wheel_angle = 0.0;
        }
    }

    if(steering_wheel_angle > max_angle_)
    {
        steering_wheel_angle = max_angle_;
    }
    else if(steering_wheel_angle < min_angle_)
    {
        steering_wheel_angle = min_angle_;
    }

    return steering_wheel_angle;

}


void YawController::setWheelBase(double wheel_base)
{
    wheel_base_ = wheel_base;
}
    
void YawController::setSteeringRatio(double steer_ratio)
{
    steer_ratio_ = steer_ratio;
}

void YawController::setParameters(double wheel_base, double steer_ratio, double max_lat_accel, double max_steer_angle)
{
    wheel_base_ = wheel_base;
    steer_ratio_ = steer_ratio;
    max_lat_accel_ = max_lat_accel;
    min_angle_ = -max_steer_angle;
    max_angle_ = max_steer_angle;
}

