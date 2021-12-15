#include <math/math_utils.hpp>
#include <SFML/Graphics/Image.hpp>
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

            Ray ray = computeRayForPixel(i, j);
            std::pair<AGeomerty *, double> closestShape = findNearestIntersect(ray); // Shape and distance to shape
            if (closestShape.first)
                rayTraceResult = computeColorFromLight(closestShape, ray);
            data.frameBuffer.set(i, j, rayTraceResult);
        }
    }
}

Color ApplicationLogic::computeColorFromLight(std::pair<AGeomerty *, double> intersectedShape, Ray & ray) {
    Vector<double ,3> color_result;
    Color result;

    /* Apply Ambient */
    auto ambient = data.m_sceneComponents.ambientLight;
    color_result[0] = std::min((ambient.color[0] * ambient.ratio), intersectedShape.first->color[0]);
    color_result[1] = std::min((ambient.color[1] * ambient.ratio), intersectedShape.first->color[1]);
    color_result[2] = std::min((ambient.color[2] * ambient.ratio), intersectedShape.first->color[2]);

    /* Apply LightSource */
    double dist = intersectedShape.second;
    Vec3d intersectionPoint = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized() * dist + ray.Origin();
    Vec3d viewDir = Vec3d::vectorFromPoints(ray.Origin(), intersectionPoint).normalized().inverse();
    Vec3d shapeNormal = intersectedShape.first->getNormalInPoint(intersectionPoint, viewDir);

    for (auto &light: data.m_sceneComponents.lights) {

        if (!lightAvailable(intersectedShape, intersectionPoint, light))
            continue;
        Vec3d lightDir = Vec3d::vectorFromPoints(intersectionPoint, light.position).normalized();

        if (dot(shapeNormal, lightDir) <= 0.0)
            continue;

        Vec3d diffuse_color;
        diffuse_color[0] = std::min(light.color[0], intersectedShape.first->color[0]);
        diffuse_color[1] = std::min(light.color[1], intersectedShape.first->color[1]);
        diffuse_color[2] = std::min(light.color[2], intersectedShape.first->color[2]);

        diffuse_color = diffuse_color * std::max((double)dot(lightDir, shapeNormal), 0.0) * light.ratio;
        color_result = color_result + diffuse_color;
    }
    color_result = clamp(color_result, 0.0, 255.0);
    result[0] = color_result[0]; result[1] = color_result[1]; result[2] = color_result[2];

    return result;
}

std::pair<AGeomerty *, double> ApplicationLogic::findNearestIntersect(const Ray &ray) {
    std::pair<AGeomerty *, double> result = {0, 100000.0};
    for (auto &geometry: data.m_sceneComponents.geometry) {
        auto distance = geometry->intersect(ray);
        if (distance.has_value() && *distance < result.second)
            result = {geometry.get(), *distance};
    }
    return result;
}

/* Screen space to scene space */
Ray ApplicationLogic::computeRayForPixel(unsigned int x, unsigned int y) {
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

void ApplicationLogic::swapFrameBufferToSfImage(sf::Image & image) {
    image.create(getFrameBuffer().width, getFrameBuffer().height, sf::Color::Black);

    /* Set image data */
    for (int i = 0; i < getFrameBuffer().width; ++i) {
        for (int j = 0; j < getFrameBuffer().height; ++j) {
            Color fb_color = data.frameBuffer.get(i, j);
            sf::Color color(fb_color[0], fb_color[1], fb_color[2]);
            image.setPixel(i, j, color);
        }
    }
}

bool ApplicationLogic::lightAvailable(std::pair<AGeomerty *, double> intersectedShape, Vec3d intersectionPoint, LightSource &light) {
    double distanceToLight = Vec3d::vectorFromPoints(light.position,intersectionPoint).length();
    Ray ray(light.position, intersectionPoint);

    std::pair<AGeomerty *, double> closestShape = findNearestIntersect(ray);
    if (closestShape.second < distanceToLight)
        return closestShape.first == intersectedShape.first;
    return true;
}
