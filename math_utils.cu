#include "math_utils.cuh"

__device__ __host__ double cuToRadian(double degree) {
    double pi = 3.14159265359;
    return (degree * (pi / 180));
}

__device__ __host__ double cuToDeg(double rad) {
    return rad * (180.0 / 3.1415);
}

__device__ __host__ cuVec3d cuGetAngles(cuVec3d v1) {
    double p_adj = sqrt(pow(v1[0], 2) + pow(v1[2], 2));
    double alpha = cuToDeg(atan2(p_adj, v1[1])); // p_adj - z axis?
    double beta = cuToDeg(atan2(v1[0], v1[2]));
    return { alpha, beta, 0.0 };
}
