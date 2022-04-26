#ifndef CUDA_LIGHT
#define CUDA_LIGHT

#include <cuda_runtime.h>
#include "Vector.cuh"

class cuLight {
public:
	void* t;
	cuColor color;
	double ratio;
	cuVec3d position;
};

class cuAmbientLight {
public:
	void* t;
	cuColor color;
	double ratio;
};


#endif