#ifndef CUDA_TRIANGLE
#define CUDA_TRIANGLE

#include <cuda_runtime.h>
#include <utility>
#include "Vector.cuh"
#include "math_utils.cuh"
#include "Ray.cuh"
#include "cuPair.cuh"
#include "AGeometry.cuh"

class cuTriangle : public cuAGeometry {
public:
    __device__ __host__ cuTriangle();
    __device__ __host__ cuTriangle(cuColor color_, cuVertex a, cuVertex b, cuVertex c);
    __device__ __host__ void print();

    cuVertex vertexes[3];

    __device__ __host__ cuPair<double, bool> intersect(const cuRay& ray);
    __device__ __host__ cuVec3d getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist);
};

#endif // !CUDA_Triangle

