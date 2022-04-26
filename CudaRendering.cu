#include "CudaRendering.cuh"
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cuPair.cuh"
#include "cudaDeviceDataPrint.cuh"
#include "CudaAppData.cuh"
#include "cuIntersection.cuh"
#include <iostream>

struct Lock {
	int* mutex;
	Lock(int count) {
		cudaMalloc(&mutex, sizeof(int) * count);
		cudaMemset(mutex,0 , sizeof(int) * count);
	}
	~Lock() {
		cudaFree(mutex);
	}

	__device__ void lock(int idx) {
		while (atomicCAS(mutex + idx, 0, 1) != 0);
	}

	__device__ void unlock(int idx) {
		atomicExch(mutex + idx, 0);
	}
};


void copyFramebufferFromGpuToCpu(void* dev_framebuffer, void* host_framebuffer, size_t data_size) {
	cudaMemcpy(host_framebuffer, dev_framebuffer, data_size, cudaMemcpyDeviceToHost);
}

__global__ void generateExampleFramebuffer(cuSceneComponents* m_sceneComponents, void* dev_data, int x, int y) {
	cuColor* colors = (cuColor*)dev_data;
	size_t pix_idx = blockIdx.x + blockIdx.y * gridDim.x;
	(colors[pix_idx])[0] = ((double)blockIdx.x / (double)gridDim.x) * 255.0;
	(colors[pix_idx])[1] = ((double)blockIdx.y / (double)gridDim.y) * 255.0;
	(colors[pix_idx])[2] = ((colors[pix_idx])[0] + (colors[pix_idx])[1]) / 2.0;
}

__device__ void printLight(const cuLight& light) {
	printVector(light.position, "light position");
	printVector(light.color, "light sphere");
	printf("light ratio: %f", light.ratio);
}

__global__ void generateRays(cuSceneComponents* m_sceneComponents, void* dev_intersectionBuffer) {
	/*
	Calculate ray (Rotate and translate ray )
	*/
	int x = blockIdx.x;
	int y = blockIdx.y;
	int w = gridDim.x;
	int h = gridDim.y;

	size_t pix_idx = blockIdx.x + blockIdx.y * gridDim.x;

	cuCamera& camera = m_sceneComponents->m_camera;

	auto ray = camera.rayForPixel(x, y, w, h);
	
	cuIntersection* intersections = (cuIntersection*)dev_intersectionBuffer;

	intersections[pix_idx].ray = ray;
}

__global__ void findIntersections(cuSceneComponents* m_sceneComponents, void* dev_intersectionBuffer, Lock lock, int w, int h) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int tid = threadIdx.x;

	size_t pix_idx = blockIdx.x + blockIdx.y * gridDim.x;
	cuIntersection* intersections = (cuIntersection*)dev_intersectionBuffer;
	cuIntersection& currentIntersection = intersections[pix_idx];

	auto ray = currentIntersection.ray;

	/*
	Iterate all shapes and try to find intersection with some, remember closest intersection result
	*/
	auto& geometry = m_sceneComponents->m_geometry.arrayPtrs;
	__shared__ cuIntersection intersection;

	intersection.distance = 100000.0;
	intersection.geometry = nullptr;
	__syncthreads();

	auto tmp_intersection = geometry[tid]->intersect(ray);
	__syncthreads();

	bool isSet = false;
	do
	{
		if (isSet = atomicCAS(lock.mutex + pix_idx, 0, 1) == 0)
		{
			// critical section goes here
			if (tmp_intersection.second && tmp_intersection.first < intersection.distance) {
				intersection.distance = tmp_intersection.first;
				intersection.geometry = geometry[tid];
			}
		}
		if (isSet)
		{
			lock.mutex[pix_idx] = 0;
		}
	} while (!isSet);

	__syncthreads();

	/*
	* Write intersection to intersection buffer
	*/
	if (tid == 0)
		intersections[pix_idx] = intersection;
}

__device__ cuIntersection findNearestIntersect(cuSceneComponents* m_sceneComponents, const cuRay& ray) {
	auto& geometry = m_sceneComponents->m_geometry;
	auto& geometry_count = m_sceneComponents->m_geometry.geometry_count;
	cuIntersection result;
	result.distance = 100000.0;
	result.geometry = nullptr;
	result.ray = ray;

	for (size_t i = 0; i < geometry_count; i++)
	{
		auto distance = geometry.arrayPtrs[i]->intersect(ray);
		if (distance.second && distance.first < result.distance) {
			result.distance = distance.second;
			result.geometry = geometry.arrayPtrs[i];
		}
	}

	return result;
}

__device__ bool cuLightAvailable(cuIntersection intersectedShape, cuSceneComponents* m_sceneComponents, cuVec3d intersectionPoint, cuLight& light) {
	double distanceToLight = cuVec3d::vectorFromPoints(light.position, intersectionPoint).length();
	cuRay ray(light.position, intersectionPoint);

	cuIntersection closestShape = findNearestIntersect(m_sceneComponents, ray);
	if (closestShape.geometry != intersectedShape.geometry) {
		return false;
	}
	return true;
}


__global__ void applyLight(cuSceneComponents* m_sceneComponents, void* dev_framebuffer, void* dev_intersectionBuffer) {

	size_t w = gridDim.x;
	size_t h = gridDim.y;
	int x = blockIdx.x;
	int y = blockIdx.y;
	size_t pix_idx = blockIdx.x + blockIdx.y * gridDim.x;
	size_t tid = threadIdx.x;
	int threads = blockDim.x;
	__shared__ cuVector<int, 3> result;
	__shared__ cuColor ambientLight;

	result[0] = 0;
	result[1] = 0;
	result[2] = 0;
	ambientLight[0] = 0.0;
	ambientLight[1] = 0.0;
	ambientLight[2] = 0.0;

	cuIntersection* intersections = (cuIntersection*)dev_intersectionBuffer;
	cuColor* framebuffer = (cuColor*)dev_framebuffer;
	auto& intersected_shape = intersections[pix_idx].geometry;

	__syncthreads();

	if (intersections[pix_idx].geometry == nullptr) {
		framebuffer[pix_idx][0] = (double)0;
		framebuffer[pix_idx][1] = (double)0;
		framebuffer[pix_idx][2] = (double)0;
		return;
	}


	/*
	* Apply ambient light
	*/

	if (tid == 0) {
		auto ambient = m_sceneComponents->m_ambientLight;
		ambientLight[0] = cuMin((ambient.color[0] * ambient.ratio), intersected_shape->color[0]);
		ambientLight[1] = cuMin((ambient.color[1] * ambient.ratio), intersected_shape->color[1]);
		ambientLight[2] = cuMin((ambient.color[2] * ambient.ratio), intersected_shape->color[2]);

		result[0] += (int)ambientLight[0];
		result[1] += (int)ambientLight[1];
		result[2] += (int)ambientLight[2];
	}
	else {
		auto& light = m_sceneComponents->m_lights[tid - 1];

		__syncthreads();

		auto ray = intersections[pix_idx].ray;
		double dist = intersections[pix_idx].distance;
		cuVec3d intersectionPoint = cuVec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized() * dist + ray.Origin();
		cuVec3d viewDir = cuVec3d::vectorFromPoints(ray.Origin(), intersectionPoint).normalized().inverse();
		cuVec3d shapeNormal = intersected_shape->getNormalInPoint(intersectionPoint, viewDir, ray, dist);

		if (!cuLightAvailable(intersections[pix_idx], m_sceneComponents, intersectionPoint, light))
			return;

		cuVec3d lightDir = cuVec3d::vectorFromPoints(intersectionPoint, light.position).normalized();

		if (cuDot(shapeNormal, lightDir) <= 0.0)
			return;
		cuColor diffuse_color;
		cuColor specular_color;

		/* Init colors */
		diffuse_color[0] = cuMin(light.color[0], intersected_shape->color[0]); /* R */
		diffuse_color[1] = cuMin(light.color[1], intersected_shape->color[1]); /* G */
		diffuse_color[2] = cuMin(light.color[2], intersected_shape->color[2]); /* B */

		/* Diffuse */
		diffuse_color = diffuse_color * cuMax((double)cuDot(lightDir, shapeNormal), 0.0) * light.ratio;

		__syncthreads();

		atomicAdd(&(result[0]), (int)diffuse_color[0]);
		atomicAdd(&(result[1]), (int)diffuse_color[1]);
		atomicAdd(&(result[2]), (int)diffuse_color[2]);
		
	}
	__syncthreads();

	if (tid == 0) {
		result = cuVecClamp(result, 0, 255);
		framebuffer[pix_idx][0] = (double)result[0];
		framebuffer[pix_idx][1] = (double)result[1];
		framebuffer[pix_idx][2] = (double)result[2];
	}
	
}

void startCudaRender(cuAppData* cuData, int w, int h) {
	dim3 grid(w, h);
	bool debug_render = false;
	auto m_sceneComponents = cuData->m_sceneComponents;
	auto dev_framebuffer = cuData->m_deviceFrameBuffer;
	auto dev_intersectionbuffer = cuData->m_deviceIntersectionBuffer;

	if (debug_render) {
		generateExampleFramebuffer << <grid, 1 >> > (m_sceneComponents, dev_intersectionbuffer, w, h);
		return;
	}
	
	Lock locks(w * h);

	generateRays << <grid, 1 >> > (m_sceneComponents, dev_intersectionbuffer);
	cudaDeviceSynchronize();

	findIntersections << <grid, cuData->m_geometryCount >> > (m_sceneComponents, dev_intersectionbuffer, locks, w, h);
	cudaDeviceSynchronize();

	applyLight <<< grid, cuData->m_lightSourcesCount + 1>>> (m_sceneComponents, dev_framebuffer, dev_intersectionbuffer);
	cudaDeviceSynchronize();

	printf("Render finished.\n");
}