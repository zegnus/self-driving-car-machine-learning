#include "utils.hpp"

namespace utils {

    double deg2rad(double x) { return x * pi() / 180; }

    double rad2deg(double x) { return x * 180 / pi(); }

    // Transform from Frenet s,d coordinates to Cartesian x,y
    std::vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y) {
        int prev_wp = -1;

        while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
        {
            prev_wp++;
        }

        int wp2 = (prev_wp+1)%maps_x.size();

        double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
        // the x,y,s along the segment
        double seg_s = (s-maps_s[prev_wp]);

        double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
        double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

        double perp_heading = heading-pi()/2;

        double x = seg_x + d*cos(perp_heading);
        double y = seg_y + d*sin(perp_heading);

        return {x,y};

    }
}