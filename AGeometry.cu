#include "AGeometry.cuh"

__device__ __host__ cuAGeometry::cuAGeometry(const cuColor& color_) : color(color_) { }

__device__ __host__ cuAGeometry::~cuAGeometry() {
}
