#ifndef CUDA_INTERSECTION
#define CUDA_INTERSECTION

#include "AGeometry.cuh"

struct cuIntersection {
	__host__ __device__ cuIntersection() : distance(100000.0), geometry(nullptr) { }

	__host__ __device__ cuIntersection(double distance_, cuAGeometry* geometry_) :
		distance(distance_), geometry(geometry_) { }

	__host__ __device__  cuIntersection(const cuIntersection& other) :
		distance(other.distance), geometry(other.geometry) { }

	__host__ __device__ cuIntersection& operator=(const cuIntersection& other) {
		if (&other == this)
			return *this;
		distance = other.distance;
		geometry = other.geometry;
	}

	cuAGeometry* geometry;
	cuRay ray;
	double distance;
};

#endif