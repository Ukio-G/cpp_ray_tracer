#include "math/math_utils.hpp"

double toRadian(double degree) {
    double pi = 3.14159265359;
    return (degree * (pi / 180));
}

double toDeg(double rad) {
    return rad * (180.0 / M_PI);
}

Vec3d getAngles(Vec3d v1) {
    double p_adj = sqrt(pow(v1[0], 2) + pow(v1[2], 2));
    double alpha = toDeg(atan2(p_adj, v1[1])); // p_adj - z axis?
    double beta = toDeg(atan2(v1[0], v1[2]));
    return {alpha, beta, 0.0};
}
