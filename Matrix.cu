#include "Matrix.cuh"
#include "Vector.cuh"
#include <math.h>

__device__ __host__ cuVec3d cuMatrix3x3::operator*(cuVec3d& cuVector) {
    double a = cuVector[0] * data[0][0] + cuVector[1] * data[0][1] + cuVector[2] * data[0][2];
    double b = cuVector[0] * data[1][0] + cuVector[1] * data[1][1] + cuVector[2] * data[1][2];
    double c = cuVector[0] * data[2][0] + cuVector[1] * data[2][1] + cuVector[2] * data[2][2];

    cuVec3d result(a, b, c);
    return result;
}

__device__ __host__ cuVec3d cuMatrix3x3::operator/(cuVec3d& cuVector) {
    cuVec3d inv(1.0 / cuVector[0], 1.0 / cuVector[1], 1.0 / cuVector[2]);
    return (*this * inv);
}

__device__ __host__ cuMatrix3x3 cuMatrix3x3::operator+(cuMatrix3x3& other) {
    cuMatrix3x3 result = *this;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result.data[i][j] += other.data[i][j];
    return result;
}

__device__ __host__ cuMatrix3x3::cuMatrix3x3(const cuVec3d& r0, const cuVec3d& r1, const cuVec3d& r2) {
    data[0][0] = r0[0];
    data[0][1] = r0[1];
    data[0][2] = r0[2];
               
    data[1][0] = r1[0];
    data[1][1] = r1[1];
    data[1][2] = r1[2];
               
    data[2][0] = r2[0];
    data[2][1] = r2[1];
    data[2][2] = r2[2];
}

__device__ __host__ cuMatrix3x3::cuMatrix3x3(const cuMatrix3x3& other) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            data[i][j] = other.data[i][j];
}

__device__ __host__ cuMatrix3x3::cuMatrix3x3() {

}

__device__ __host__ cuMatrix3x3 cuMatrix3x3::rotateX(double alpha) {
    cuMatrix3x3 result;

    result.data[0][0] = 1;
    result.data[0][1] = 0;
    result.data[0][2] = 0;
    result.data[1][0] = 0;
    result.data[1][1] = cos(alpha);
    result.data[1][2] = -sin(alpha);
    result.data[2][0] = 0;
    result.data[2][1] = sin(alpha);
    result.data[2][2] = cos(alpha);

    return result;
}

__device__ __host__ cuMatrix3x3 cuMatrix3x3::rotateY(double beta) {
    cuMatrix3x3 result;

    result.data[0][0] = cos(beta);
    result.data[0][1] = 0;
    result.data[0][2] = sin(beta);
    result.data[1][0] = 0;
    result.data[1][1] = 1;
    result.data[1][2] = 0;
    result.data[2][0] = -sin(beta);
    result.data[2][1] = 0;
    result.data[2][2] = cos(beta);

    return result;
}

__device__ __host__ cuMatrix3x3 cuMatrix3x3::rotateZ(double gamma) {
    cuMatrix3x3 result;

    result.data[0][0] = cos(gamma);
    result.data[0][1] = -sin(gamma);
    result.data[0][2] = 0;
    result.data[1][0] = sin(gamma);
    result.data[1][1] = cos(gamma);
    result.data[1][2] = 0;
    result.data[2][0] = 0;
    result.data[2][1] = 0;
    result.data[2][2] = 1;

    return result;
}


