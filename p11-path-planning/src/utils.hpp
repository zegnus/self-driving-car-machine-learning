#include <math.h>
#include <vector>

using namespace std;

namespace utils {

    constexpr double pi() { return M_PI; }

    double deg2rad(double x);
    
    double rad2deg(double x);

    vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y);
}