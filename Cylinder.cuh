#ifndef CUDA_CYLINDER
#define CUDA_CYLINDER
#include <cuda_runtime.h>
#include <utility>
#include "Vector.cuh"
#include "math_utils.cuh"
#include "Ray.cuh"
#include "cuPair.cuh"
#include "AGeometry.cuh"
#include "Matrix.cuh"

class cuCylinder : public cuAGeometry {
public:
    __host__ __device__ cuCylinder();
    __host__ __device__ cuCylinder(cuColor color_, double diameter_, double height_, cuVec3d position_, cuVec3d direction_);
    __device__ __host__ void print();

    double diameter;
    double radius;
    double height;

    // Bound box of cylinder (for reduce time intersection calculating)
    double boundBox[8];

    cuVec3d position;
    cuVec3d direction;

private:
    cuVec3d BBoxMin;
    cuVec3d BBoxMax;
    cuVec3d _bottomPoint;
    cuVec3d _topPoint;

    __host__ __device__ cuPair<double, bool> intersect(const cuRay& ray);
    __host__ __device__ cuVec3d getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist);

    __host__ __device__ cuPair<double, bool> checkCandidate(cuVec3d originRay, cuVec3d rayDirection, double dist);
};

#endif // !CUDA_CYLINDER