#ifndef CUDA_RENDERING
#define CUDA_RENDERING


/*
	Define only public API for calling from CPU
*/

#include <cuda_runtime.h>
#include "CudaSceneComponents.cuh"
#include "CudaAppData.cuh"


void copyFramebufferFromGpuToCpu(void* dev_framebuffer, void* host_framebuffer, size_t data_size);
void startCudaRender(cuAppData* cuData, int x, int y);




#endif