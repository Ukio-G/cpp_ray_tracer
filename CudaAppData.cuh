#ifndef CUDA_APP_DATA
#define CUDA_APP_DATA

#include <cuda_runtime.h>
#include "CudaGeometryCollection.cuh"
#include "CudaInitStruct.h"
#include "CudaSceneComponents.cuh"

class cuAppData {
public:
	__host__ cuAppData() { } ;
	__host__ void generatePointersArray(cuGeometryCollection& host_cuGeometry);
	__host__ cuAppData(CudaInitStruct* initStruct);

//private:
	cuSceneComponents* m_sceneComponents = 0;	/* DEVICE MEMORY */
	void* m_deviceFrameBuffer = 0;				/* DEVICE MEMORY 24 bytes per block */
	void* m_deviceIntersectionBuffer = 0;       /* DEVICE MEMORY 16 bytes per block */
	int  m_geometryCount = 0;					/* HOST MEMORY */
	int  m_lightSourcesCount = 0;				/* HOST MEMORY */


};

#endif // !CUDA_APP_DATA
