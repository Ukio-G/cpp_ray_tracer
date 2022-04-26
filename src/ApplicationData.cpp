#include "ApplicationData.hpp"

CudaInitStruct ApplicationData::generateCudaInitStruct()
{
	CudaInitStruct result;

	result.lights       = &m_sceneComponents.lights;
	result.ambientLight = &m_sceneComponents.ambientLight;

	memcpy(&result.cameraPosition, &m_sceneComponents.cameras[0].position, sizeof(Vec3d));
	memcpy(&result.cameraDirection, &m_sceneComponents.cameras[0].direction, sizeof(Vec3d));
	result.cameraFov = m_sceneComponents.cameras[0].fov;

	result.framebufferSize[0] = frameBuffer.width;
	result.framebufferSize[1] = frameBuffer.height;

	result.continuousData = m_geometryGenerator.getData();
	result.continuousBytesCount = m_geometryGenerator.getSize();

	result.m_shapesCount = m_geometryCounter.getShapesCount();

	result.m_spheresCount = m_geometryCounter.getSpheresCount();
	result.m_planesCount = m_geometryCounter.getPlanesCount();
	result.m_cylindersCount = m_geometryCounter.getCylindersCount();
	result.m_trianglesCount = m_geometryCounter.getTrianglesCount();
	result.m_squaresCount = m_geometryCounter.getSquaresCount();

	result.m_spheres_begin_ptr   = m_geometryGenerator.getSpheresPtr();
	result.m_planes_begin_ptr    = m_geometryGenerator.getPlanesPtr();
	result.m_cylinders_begin_ptr = m_geometryGenerator.getCylindersPtr();
	result.m_triangles_begin_ptr = m_geometryGenerator.getTrianglesPtr();
	result.m_squares_begin_ptr   = m_geometryGenerator.getSquaresPtr();

	return result;
}