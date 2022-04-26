#ifndef APPLICATIONDATA_HPP
#define APPLICATIONDATA_HPP

#include <vector>
#include "geometry/AGeomerty.hpp"
#include "SceneComponents.hpp"
#include "Parser.hpp"
#include "FrameBuffer.hpp"
#include <functional>
#include "../continuousGeometryGenerator.h"
#include "../GeometryCounter.h"
#include "../CudaInitStruct.h"

class ApplicationLogic;
class cuAppData;

class ApplicationData {
    friend ApplicationLogic;
    friend cuAppData;
public:
    ApplicationData() = default;

    CudaInitStruct generateCudaInitStruct();
// private:
    /*
    ** Counter of geometry inside scene configuration file.
    *  Used for geometryGenerator, as reference to memory allocation
    */
    GeometryCounter m_geometryCounter;

    /*
    ** Continuous memory allocator for scene's geometry.
    *  Keep all data isdide continuous memory block
    *  Used for CUDA Memcpy (We want only 1 copy from CPU RAM to GPU RAM)
    */
    ContinuousGeometryGenerator m_geometryGenerator;

    SceneComponents m_sceneComponents;
    double m_projectionPlaneDistance = 0;
    int currentCameraIdx;
    Parser parser;
    FrameBuffer frameBuffer;
};

#endif //APPLICATIONDATA_HPP
