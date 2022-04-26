#include "CudaAppData.cuh"
#include "LightSource.hpp"
#include "Camera.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuPair.cuh"
#include "cudaDeviceDataPrint.cuh"
#include "cuIntersection.cuh"
#include <iostream>

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

__device__ void copyVTable(void* dst, void* src) {
	uint64_t* dst_u64 = (uint64_t*)dst;
	uint64_t* src_u64 = (uint64_t*)src;
#ifdef DEBUG_CUDA_PRINT
	printf("copy vtabe: was %p, now:%p\n", (void*)dst_u64[0], (void*)src_u64[0]);
#endif
	dst_u64[0] = src_u64[0];
}

__global__ void createVTablesForGeometry(cuSceneComponents* sceneComponents) {
	auto spheres = sceneComponents->m_geometry.m_spheres_begin_ptr;
	auto planes = sceneComponents->m_geometry.m_planes_begin_ptr;
	auto squares = sceneComponents->m_geometry.m_squares_begin_ptr;
	auto triangles = sceneComponents->m_geometry.m_triangles_begin_ptr;
	auto cylinders = sceneComponents->m_geometry.m_cylinders_begin_ptr;

	cuSphere	sphere;
	cuPlane		plane;
	cuSquare	square;
	cuTriangle	triangle;
	cuCylinder	cylinder;

	for (size_t i = 0; i < sceneComponents->m_geometry.spheres_count; i++)
		copyVTable(sceneComponents->m_geometry.m_spheres_begin_ptr + i, &sphere);

	for (size_t i = 0; i < sceneComponents->m_geometry.planes_count; i++)
		copyVTable(sceneComponents->m_geometry.m_planes_begin_ptr + i, &plane);

	for (size_t i = 0; i < sceneComponents->m_geometry.squares_count; i++)
		copyVTable(sceneComponents->m_geometry.m_squares_begin_ptr + i, &square);

	for (size_t i = 0; i < sceneComponents->m_geometry.triangles_count; i++)
		copyVTable(sceneComponents->m_geometry.m_triangles_begin_ptr + i, &triangle);

	for (size_t i = 0; i < sceneComponents->m_geometry.cylinders_count; i++)
		copyVTable(sceneComponents->m_geometry.m_cylinders_begin_ptr + i, &cylinder);
#ifdef DEBUG_CUDA_PRINT
	printComponents(sceneComponents);
#endif
}

__host__ void cuAppData::generatePointersArray(cuGeometryCollection & host_cuGeometry) {
	auto dev_arrayPtrs = host_cuGeometry.arrayPtrs;
	std::vector<cuAGeometry*> tmp_arrayPtrs;

	//Spheres
	for (size_t i = 0; i < host_cuGeometry.spheres_count; i++)
		tmp_arrayPtrs.push_back(host_cuGeometry.m_spheres_begin_ptr + i);

	//Squares
	for (size_t i = 0; i < host_cuGeometry.squares_count; i++)
		tmp_arrayPtrs.push_back(host_cuGeometry.m_squares_begin_ptr + i);

	//Cylindres
	for (size_t i = 0; i < host_cuGeometry.cylinders_count; i++)
		tmp_arrayPtrs.push_back(host_cuGeometry.m_cylinders_begin_ptr + i);

	//Triangles
	for (size_t i = 0; i < host_cuGeometry.triangles_count; i++)
		tmp_arrayPtrs.push_back(host_cuGeometry.m_triangles_begin_ptr + i);

	//Planes
	for (size_t i = 0; i < host_cuGeometry.planes_count; i++)
		tmp_arrayPtrs.push_back(host_cuGeometry.m_planes_begin_ptr + i);

	cudaMemcpy(dev_arrayPtrs, tmp_arrayPtrs.data(), tmp_arrayPtrs.size() * sizeof(cuAGeometry*), cudaMemcpyHostToDevice);
#ifdef DEBUG_CUDA_PRINT

	printf("m_spheres_begin_ptr: %p\n", host_cuGeometry.m_spheres_begin_ptr);
	printf("m_squares_begin_ptr: %p\n", host_cuGeometry.m_squares_begin_ptr);
	printf("m_cylinders_begin_ptr: %p\n", host_cuGeometry.m_cylinders_begin_ptr);
	printf("m_triangles_begin_ptr: %p\n", host_cuGeometry.m_triangles_begin_ptr);
	printf("m_planes_begin_ptr: %p\n", host_cuGeometry.m_planes_begin_ptr);


	for (size_t i = 0; i < tmp_arrayPtrs.size(); i++)
	{
		printf("tmp_arrayPtrs[%i]: %p\n", i, tmp_arrayPtrs[i]);
	}
#endif
}

__host__ cuAppData::cuAppData(CudaInitStruct* initStruct) {
	cuSceneComponents sceneComponents;

	void* newData = 0;

	// size in bytes
	size_t geometry_size    = initStruct->continuousBytesCount;
	size_t spheres_bytes	= initStruct->m_spheresCount   * sizeof(cuSphere);
	size_t squares_bytes	= initStruct->m_squaresCount   * sizeof(cuSquare);
	size_t cylinders_bytes  = initStruct->m_cylindersCount * sizeof(cuCylinder);
	size_t planes_bytes		= initStruct->m_planesCount	   * sizeof(cuPlane);
	size_t triangles_bytes	= initStruct->m_trianglesCount * sizeof(cuTriangle);

	// Allocate and copy geometry
	cudaMalloc(&newData, geometry_size);
	cudaMemcpy(newData, initStruct->continuousData, geometry_size, cudaMemcpyHostToDevice);
	auto& cuGeometry = sceneComponents.m_geometry;

	cuGeometry.data = (cuAGeometry*)newData;
	cuGeometry.m_spheres_begin_ptr   = (cuSphere*)newData;
	cuGeometry.m_squares_begin_ptr   = (cuSquare*)((unsigned char*)cuGeometry.m_spheres_begin_ptr   + spheres_bytes);
	cuGeometry.m_cylinders_begin_ptr = (cuCylinder*)((unsigned char*)cuGeometry.m_squares_begin_ptr + squares_bytes);
	cuGeometry.m_planes_begin_ptr    = (cuPlane*)((unsigned char*)cuGeometry.m_cylinders_begin_ptr  + cylinders_bytes);
	cuGeometry.m_triangles_begin_ptr = (cuTriangle*)((unsigned char*)cuGeometry.m_planes_begin_ptr  + planes_bytes);

	cuGeometry.spheres_count   = initStruct->m_spheresCount;
	cuGeometry.squares_count   = initStruct->m_squaresCount;
	cuGeometry.cylinders_count = initStruct->m_cylindersCount;
	cuGeometry.planes_count    = initStruct->m_planesCount;
	cuGeometry.triangles_count = initStruct->m_trianglesCount;

	cuGeometry.geometry_size = geometry_size;

	m_geometryCount = initStruct->m_shapesCount;
	cuGeometry.geometry_count = m_geometryCount;
	cudaMalloc(&(cuGeometry.arrayPtrs), m_geometryCount * sizeof(cuAGeometry*));
	generatePointersArray(cuGeometry);

	// Allocate and copy camera
	sceneComponents.m_camera.position = initStruct->cameraPosition;
	sceneComponents.m_camera.direction = initStruct->cameraDirection;
	sceneComponents.m_camera.fov = initStruct->cameraFov;

	// Allocate and copy light sources
	
	// size in bytes
	size_t lights_size = initStruct->lights->size() * sizeof(LightSource);
	cudaMalloc(&sceneComponents.m_lights, lights_size);
	cudaMemcpy(sceneComponents.m_lights, initStruct->lights->data(), lights_size, cudaMemcpyHostToDevice);

	for (size_t i = 0; i < 3; i++)
		sceneComponents.m_ambientLight.color[i] = initStruct->ambientLight->color[i];
	sceneComponents.m_ambientLight.ratio = initStruct->ambientLight->ratio;
	m_lightSourcesCount = initStruct->lights->size();

	
	// Generate frame and intersections buffers
	sceneComponents.m_frameBufferSize[0] = initStruct->framebufferSize[0];
	sceneComponents.m_frameBufferSize[1] = initStruct->framebufferSize[1];

	sceneComponents.m_projectionPlaneDistance = initStruct->projectionPlaneDistance;

	cudaMalloc(&m_sceneComponents, sizeof(cuSceneComponents));
	cudaMemcpy(m_sceneComponents, &sceneComponents, sizeof(cuSceneComponents), cudaMemcpyHostToDevice);

	auto pixels_count = initStruct->framebufferSize[0] * initStruct->framebufferSize[1];

	HANDLE_ERROR(cudaMalloc(&m_deviceFrameBuffer, pixels_count * sizeof(cuColor)));
	HANDLE_ERROR(cudaMalloc(&m_deviceIntersectionBuffer, pixels_count * sizeof(cuIntersection) ));
	createVTablesForGeometry << <1, 1 >> > (m_sceneComponents);

#ifdef DEBUG_CUDA_PRINT
	printMemoryBlock(initStruct->continuousData, geometry_size);
#endif
}