#ifndef CUDA_SCENE_COMPONENTS
#define CUDA_SCENE_COMPONENTS

#include <cuda_runtime.h>
#include "CudaGeometryCollection.cuh"
#include "Camera.cuh"
#include "Light.cuh"


class cuSceneComponents {
public:
	cuGeometryCollection m_geometry;
	
	cuCamera             m_camera;
	double				 m_projectionPlaneDistance = 0.0;

	cuLight*             m_lights = 0;
	size_t               m_lightsCount = 0;

	cuAmbientLight       m_ambientLight;

	cuVec2i              m_frameBufferSize = {0,0}; // (width, height) grid dimensions

	void*				result = 0;
	size_t				result_size = 0;
};


#endif // !CUDA_SCENE_COMPONENTS
