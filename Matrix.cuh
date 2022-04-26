#ifndef CUDA_MATRIX
#define CUDA_MATRIX

#include <cuda_runtime.h>
#include "Vector.cuh"

class cuMatrix3x3 {
public:
    double data[3][3];
    __device__ __host__ cuMatrix3x3();

    /* Vectors are rows */
    __device__ __host__ cuMatrix3x3(const cuVec3d& r0, const cuVec3d& r1, const cuVec3d& r2);

    __device__ __host__ cuMatrix3x3(const cuMatrix3x3& other);
    __device__ __host__ cuVec3d operator*(cuVec3d& vector);
    __device__ __host__ cuVec3d operator/(cuVec3d& vector);
    __device__ __host__ cuMatrix3x3 operator+(cuMatrix3x3& other);

    __device__ __host__ static cuMatrix3x3 rotateX(double alpha);
    __device__ __host__ static cuMatrix3x3 rotateY(double beta);
    __device__ __host__ static cuMatrix3x3 rotateZ(double gamma);
};

#endif