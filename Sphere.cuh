#ifndef CUDA_SPHERE
#define CUDA_SPHERE
#include <cuda_runtime.h>
#include <utility>
#include "Vector.cuh"
#include "math_utils.cuh"
#include "Ray.cuh"
#include "cuPair.cuh"
#include "AGeometry.cuh"


class cuSphere : public cuAGeometry {
public:
    __device__ __host__ cuSphere();
    __device__ __host__ cuSphere(cuColor color_, cuVec3d position_, double diameter_);

    __device__ __host__ cuPair<double, bool> intersect(const cuRay& ray);
    __device__ __host__ cuVec3d getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const  cuRay& ray, double dist);
    __device__ __host__ void print();

    cuVec3d position;
    double diameter;
    double radius;

    double m_radius_sq;
    bool m_inverse = false;
    bool m_inversable = true;
};


#endif // !CUDA_SPHERE
