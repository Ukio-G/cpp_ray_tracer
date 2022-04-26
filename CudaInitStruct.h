#pragma once
#include "Vector.cuh"
#include "LightSource.hpp"
#include "AmbientLight.hpp"
#include <vector>

struct CudaInitStruct {
public:

	/*
	Light data
	*/
	std::vector<LightSource>* lights;
	AmbientLight* ambientLight;

	/*
	Camera data
	*/
	cuVec3d cameraPosition;
	cuVec3d cameraDirection;
	double cameraFov;
	double projectionPlaneDistance;

	/*
	Framebuffer size
	*/
	cuVec2i framebufferSize;

	void* continuousData;
	size_t continuousBytesCount;
	size_t m_shapesCount;

	size_t m_spheresCount;
	size_t m_planesCount;
	size_t m_cylindersCount;
	size_t m_trianglesCount;
	size_t m_squaresCount;


	void* m_spheres_begin_ptr;
	void* m_planes_begin_ptr;
	void* m_cylinders_begin_ptr;
	void* m_triangles_begin_ptr;
	void* m_squares_begin_ptr;
};