#ifndef CUDA_RAY
#define CUDA_RAY

#include <cuda_runtime.h>
#include "Vector.cuh"


class cuRay
{
public:
    __device__ __host__ cuRay();
    __device__ __host__ cuRay(const cuVec3d& O, const cuVec3d& D);
    __device__ __host__ ~cuRay();

    __device__ __host__ cuVec3d Origin() const;
    __device__ __host__ cuVec3d Direction() const;
    __device__ __host__ cuVec3d pointAt(double t);
    int test = 0;
private:
    cuVec3d origin;
    cuVec3d direction;
};

#endif