#ifndef CUDA_AGEOMETRY
#define CUDA_AGEOMETRY

#include <cuda_runtime.h>
#include "Vector.cuh"
#include "cuPair.cuh"
#include "Ray.cuh"

class cuAGeometry {
public:
	cuColor color;
	__device__ __host__ cuAGeometry::cuAGeometry(const cuColor& color_);

	__device__ __host__ virtual ~cuAGeometry();

	__device__ __host__ virtual cuPair<double, bool> intersect(const cuRay& ray) = 0;
	__device__ __host__ virtual void print() = 0;
	__device__ __host__ virtual cuVec3d getNormalInPoint(const cuVec3d& point, const cuVec3d& view, const cuRay& ray, double dist) = 0;

};

#endif // !CUDA_SQUARE

