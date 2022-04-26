#include <stdio.h>
#include <stdlib.h>
#include "cudaDeviceDataPrint.cuh"

static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ __host__  void printCudaDevices(void) {
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf(" --- General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execition timeout : ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf(" --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf(" --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count: %d\n",
		prop.multiProcessorCount);
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n",
		prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2]);
		printf("\n");
	}


}

void printVector(const cuVec3d& vec, char* prefix) {
	printf("%s (%f, %f, %f)\n", prefix, vec[0], vec[1], vec[2]);
}

void printRay(const cuRay& ray) {
	printf("Origin: ( %f, %f, %f ), Direction: ( %f, %f, %f ) \n", ray.Origin()[0], ray.Origin()[1], ray.Origin()[2], ray.Direction()[0], ray.Direction()[1], ray.Direction()[2]);
}


void printComponents(cuSceneComponents* sceneComponents) {
	size_t geometry_count = sceneComponents->m_geometry.geometry_count;
	cuAGeometry** geometry_ptr = sceneComponents->m_geometry.arrayPtrs;

	for (size_t i = 0; i < geometry_count; i++)
	{	
		printf("Geometry #%i\n", i);
		geometry_ptr[i]->print();
		printf("=======================\n");

	}

	printMemoryBlock(sceneComponents->m_geometry.data, sceneComponents->m_geometry.geometry_size);
}


void printMemoryBlock(void* ptr, size_t count) {
	unsigned char* data = (unsigned char*)ptr;

	printf("========== Print %i bytes start from %p ==========\n", (int)count, ptr);

	for (size_t i = 0; i < count; i++)
	{
		if (i % 16 == 0) {
			if (i > 0)
				printf("\n");
			printf("%p: ", data + i);
		}
		printf("%02X ", data[i]);
	}
	printf("\n");
}