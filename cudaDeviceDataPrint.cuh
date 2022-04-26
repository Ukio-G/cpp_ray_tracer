#ifndef CUDA_PRINT_DEV
#define CUDA_PRINT_DEV
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaSceneComponents.cuh"

__device__ __host__ void printCudaDevices(void);
__device__ __host__ void printVector(const cuVec3d& vec, char* prefix);
__device__ __host__ void printRay(const cuRay& ray);
__device__ __host__ void printComponents(cuSceneComponents* sceneComponents);
__device__ __host__ void printMemoryBlock(void* ptr, size_t count);

#endif