#ifndef CUDA_SQUARE
#define CUDA_SQUARE
#include <cuda_runtime.h>
#include <utility>
#include "Vector.cuh"
#include "math_utils.cuh"
#include "Ray.cuh"
#include "cuPair.cuh"
#include "AGeometry.cuh"
#include "Matrix.cuh"

class cuSquare : public cuAGeometry {
public:
    __device__ __host__  cuSquare();
    __device__ __host__ cuSquare(cuColor color_, cuVec3d center_, cuVec3d direction_, double sizeSide_);
    __device__ __host__ void print();

    cuVec3d center;
    cuVec3d  direction;
    double sizeSide;
    cuVertex vertexes[4];

    __device__  __host__  cuPair<double, bool> intersect(const cuRay& ray);
    __device__  __host__  cuVec3d  getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist);
    __device__  __host__  void initVertexes();
};

#endif // !CUDA_SQUARE

