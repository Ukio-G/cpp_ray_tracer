#ifndef CUDA_PLANE
#define CUDA_PLANE

#include <cuda_runtime.h>
#include <utility>
#include "Vector.cuh"
#include "math_utils.cuh"
#include "Ray.cuh"
#include "cuPair.cuh"
#include "AGeometry.cuh"


class cuPlane : public cuAGeometry {
public:
	__device__ __host__ cuPlane();
	__device__ __host__ cuPlane(cuColor color_, cuVec3d position_, cuVec3d normal_);
	__device__ __host__ void print();

	cuVec3d position;
	cuVec3d normal;

	__device__ __host__ cuPair<double, bool> intersect(const cuRay& ray);
	__device__ __host__ cuVec3d getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist);
};

#endif // !CUDA_PLANE

