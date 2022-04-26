#include "Ray.cuh"

__device__ __host__ cuRay::cuRay() { };
__device__ __host__ cuRay::cuRay(const cuVec3d& O, const cuVec3d& D) : origin(O), direction(D) { };
__device__ __host__ cuRay::~cuRay() { };

__device__ __host__ cuVec3d cuRay::Origin() const { return origin; };
__device__ __host__ cuVec3d cuRay::Direction() const { return direction; };
__device__ __host__ cuVec3d cuRay::pointAt(double t) { return origin + direction * t; };