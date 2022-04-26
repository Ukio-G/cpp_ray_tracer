#ifndef SCENECOMPONENTS_HPP
#define SCENECOMPONENTS_HPP

#include <geometry/AGeomerty.hpp>
#include <memory>
#include "Camera.hpp"
#include "LightSource.hpp"
#include "AmbientLight.hpp"
#include <vector>

struct SceneComponents {
public:
    std::vector<std::shared_ptr<AGeomerty>>     geometry;
    std::vector<Camera>                         cameras;
    std::vector<LightSource>                    lights;
    void*                                       continuousGeometry;
    AmbientLight                                ambientLight;
    Vec2i                                       windowResolution;
};

#endif
