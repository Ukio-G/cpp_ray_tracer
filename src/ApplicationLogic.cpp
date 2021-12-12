#include <math/math_utils.hpp>
#include "ApplicationLogic.hpp"

ApplicationLogic::ApplicationLogic() {

}

void ApplicationLogic::initFromFile(const std::string &filename) {
    data.m_sceneComponents = data.parser.parseFile(filename);
    auto window_size = data.m_sceneComponents.windowResolution;
    data.frameBuffer = FrameBuffer(window_size[0], window_size[1]);

    if (!data.m_sceneComponents.cameras.empty())
        data.currentCameraIdx = 0;
}

const FrameBuffer &ApplicationLogic::getFrameBuffer() {
    return data.frameBuffer;
}

void ApplicationLogic::renderFrameBuffer() {
    for (int i = 0; i < data.frameBuffer.width; ++i) {
        for (int j = 0; j < data.frameBuffer.height; ++j) {
            Color rayTraceResult;

            Ray ray = computeForPixel(i, j);
            std::pair<AGeomerty *, double> closestShape = findNearestIntersect(ray); // Shape and distance to shape
            if (closestShape.first)
                rayTraceResult = computeColorFromLight(closestShape);
            data.frameBuffer.set(i, j, rayTraceResult);
        }
    }
}

Color ApplicationLogic::computeColorFromLight(std::pair<AGeomerty *, double> intersectedShape) {
    return intersectedShape.first->color;
}

std::pair<AGeomerty *, double> ApplicationLogic::findNearestIntersect(const Ray &ray) {
    std::pair<AGeomerty *, double> result = {0, 0};
    for (auto &geometry: data.m_sceneComponents.geometry) {
        auto distance = geometry->intersect(ray);
        if (distance.has_value() && distance < result.second)
            result = {geometry.get(), *distance};
    }
    return result;
}

Ray ApplicationLogic::computeForPixel(unsigned int x, unsigned int y) {
    auto & framebuffer = data.frameBuffer;
    auto & currentCamera = data.m_sceneComponents.cameras[data.currentCameraIdx];
    auto start = currentCamera.position;
    auto half_width = (double)framebuffer.width / 2.0;

    auto x_r = (double)x - half_width;
    auto y_r = (double)framebuffer.height / 2.0 - (double)y;
    auto z_r = half_width / tan(toRadian(currentCamera.fov / 2.0));

    Vec3d end(x_r, y_r, z_r);

    return Ray(start, end);
}
