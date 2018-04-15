#include "lowpassfilter.h"

using namespace DBWNODE_NS;

LowPassFilter::LowPassFilter()
{
    ready_ = false;
    last_val_ = 0.0;
    a_ = 1.0;
    b_ = 0.0;
}

LowPassFilter::LowPassFilter(double tau, double ts)
{
    last_val_ = 0.0;
    ready_ = false;

    setParams(tau, ts);
}

void LowPassFilter::setParams(double tau, double ts) 
{
    a_ = 1.0 / (tau / ts + 1.0);
    b_ = tau / ts / (tau / ts + 1.0);
}

LowPassFilter::~LowPassFilter()
{}

double LowPassFilter::filt(double val)
{
    double value = val;
    if(ready_ == true)
    {
        value = a_ * val + b_ * last_val_;
    }
    else
    {
        ready_ = true;
    }

    last_val_ = value;

    return value;
}

double LowPassFilter::get() 
{ 
    return last_val_; 
}

bool LowPassFilter::getReady() 
{ 
    return ready_; 
}