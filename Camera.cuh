#ifndef CUDA_CAMERA
#define CUDA_CAMERA

#include <cuda_runtime.h>
#include "Ray.cuh"
#include "Matrix.cuh"
#include "math_utils.cuh"

class cuCamera {
public:
    __host__ __device__ cuCamera();
    __host__ __device__ cuRay rayForPixel(unsigned int x, unsigned int y, unsigned int w, unsigned int h);

    cuVec3d position;
    cuVec3d direction;
    double fov;
};

#endif // !CUDA_CAMERA

