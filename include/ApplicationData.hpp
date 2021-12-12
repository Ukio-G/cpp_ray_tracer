#ifndef APPLICATIONDATA_HPP
#define APPLICATIONDATA_HPP

#include <vector>
#include "geometry/AGeomerty.hpp"
#include "SceneComponents.hpp"
#include "Parser.hpp"
#include "FrameBuffer.hpp"
#include <functional>

class ApplicationLogic;

class ApplicationData {
    friend ApplicationLogic;
public:
    ApplicationData() = default;
private:
    SceneComponents m_sceneComponents;
    double m_projectionPlaneDistance = 0;
    int currentCameraIdx;
    Parser parser;
    FrameBuffer frameBuffer;
};

#endif //APPLICATIONDATA_HPP
