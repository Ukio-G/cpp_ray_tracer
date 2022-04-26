#include <math/math_utils.hpp>
#include <SFML/Graphics/Image.hpp>
#include <iostream>
#include "ApplicationLogic.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <time.h>
#include <chrono>
#include <geometry/Sphere.hpp>
#include "../CudaRendering.cuh"
#include "../TimeLogger.h"


ApplicationLogic::ApplicationLogic() {

}

void ApplicationLogic::initFromFile(const std::string &filename) {
    data.m_sceneComponents = data.parser.parseFile(filename);
    auto window_size = data.m_sceneComponents.windowResolution;
    data.frameBuffer = FrameBuffer(window_size[0], window_size[1]);


    for(auto & item: data.m_sceneComponents.geometry) {
        if (auto itemPtr = dynamic_cast<Cylinder*>(item.get())) {
            auto spheres = itemPtr->initBoundBox();

            for (auto &sphere: spheres) {
                std::shared_ptr<Sphere> sphere_ptr(new Sphere(sphere));
                data.m_sceneComponents.geometry.push_back(sphere_ptr);
            }
        }
    }

    if (!data.m_sceneComponents.cameras.empty())
        data.currentCameraIdx = 0;
}

void ApplicationLogic::initFromFileContinuous(const std::string& filename)
{
    data.m_geometryCounter.countFromFile(filename);
    data.m_geometryGenerator.allocateMemory(data.m_geometryCounter);
    data.parser.parseFile(filename, data.m_geometryGenerator);
}

const FrameBuffer &ApplicationLogic::getFrameBuffer() {
    return data.frameBuffer;
}

void ApplicationLogic::renderFrameBufferCUDA() {
    size_t w = data.frameBuffer.width;
    size_t h = data.frameBuffer.height;
    
    TimeLogger::getInstance()->StartMeasure("CudaRender");
    startCudaRender(cuData, w, h);
    TimeLogger::getInstance()->StopMeasure("CudaRender");

    copyFramebufferFromGpuToCpu(cuData->m_deviceFrameBuffer, data.frameBuffer.buffer, w * h * sizeof(Color));
}


void ApplicationLogic::renderFrameBufferCPU() {
    TimeLogger::getInstance()->StartMeasure("CPURender");
    for (int i = 0; i < data.frameBuffer.width; ++i) {
        for (int j = 0; j < data.frameBuffer.height; ++j) {
            Color rayTraceResult(0.0, 0.0, 0.0);
            auto& currentCamera = data.m_sceneComponents.cameras[data.currentCameraIdx];
            Ray ray = currentCamera.computeRayForPixel(i, j, data.frameBuffer);
            std::pair<AGeomerty*, double> closestShape = findNearestIntersect(ray); // Shape and distance to shape
            if (closestShape.first)
                rayTraceResult = computeColorFromLight(closestShape, ray);
            data.frameBuffer.set(i, j, rayTraceResult);

        }
    }
    TimeLogger::getInstance()->StopMeasure("CPURender");
}

void ApplicationLogic::renderFrameBuffer(RenderMode renderMode = CPU) {
    if (renderMode == CPU)
        renderFrameBufferCPU();
    if (renderMode == CUDA)
        renderFrameBufferCUDA();
}

// TODO: Add comments
Color ApplicationLogic::computeColorFromLight(std::pair<AGeomerty *, double> intersectedShape, Ray & ray) {
    Vector<double ,3> color_result;

    /* Apply Ambient */
    auto ambient = data.m_sceneComponents.ambientLight;
    color_result[0] = std::min((ambient.color[0] * ambient.ratio), intersectedShape.first->color[0]);
    color_result[1] = std::min((ambient.color[1] * ambient.ratio), intersectedShape.first->color[1]);
    color_result[2] = std::min((ambient.color[2] * ambient.ratio), intersectedShape.first->color[2]);

    /* Apply LightSource */
    double dist = intersectedShape.second;

    Vec3d intersectionPoint = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized() * dist + ray.Origin();
    Vec3d viewDir = Vec3d::vectorFromPoints(ray.Origin(), intersectionPoint).normalized().inverse();
    Vec3d shapeNormal = intersectedShape.first->getNormalInPoint(intersectionPoint, viewDir, ray, dist);

    for (auto &light: data.m_sceneComponents.lights) {
        /* If we have intersection between intersection point and current light source - skip iteration */
        if (!lightAvailable(intersectedShape, intersectionPoint, light))
            continue;
        Vec3d lightDir = Vec3d::vectorFromPoints(intersectionPoint, light.position).normalized();

        if (dot(shapeNormal, lightDir) <= 0.0)
            continue;

        Vec3d diffuse_color;
        Vec3d specular_color;

        /* Init colors */
        diffuse_color[0] = std::min(light.color[0], intersectedShape.first->color[0]); /* R */
        diffuse_color[1] = std::min(light.color[1], intersectedShape.first->color[1]); /* G */
        diffuse_color[2] = std::min(light.color[2], intersectedShape.first->color[2]); /* B */
        specular_color = diffuse_color;

        /* Diffuse */
        diffuse_color = diffuse_color * std::max((double)dot(lightDir, shapeNormal), 0.0) * light.ratio;

        /* Specular */
        Vec3d l = Vec3d::vectorFromPoints(light.position, intersectionPoint);
        Vec3d r = reflect(l, shapeNormal).normalized();
        specular_color = specular_color * pow(std::max(dot(r, viewDir), 0.0), 128);

        /* Append Diffuse and specular to result color */
        color_result = color_result + diffuse_color + specular_color;
    }
    return clamp(color_result, 0.0, 255.0);
}

std::pair<AGeomerty *, double> ApplicationLogic::findNearestIntersect(const Ray &ray) {
    std::pair<AGeomerty *, double> result = {0, 100000.0};
    /*
    for (auto &geometry: data.m_sceneComponents.geometry) {
        auto distance = geometry->intersect(ray);
        if (distance.has_value() && *distance < result.second)
            result = {geometry.get(), *distance};
    }
    */

    auto& counter = data.m_geometryCounter;
    auto& geometry = data.m_geometryGenerator;

    auto sp_ptr = geometry.getSpheresPtr();
    auto pl_ptr = geometry.getPlanesPtr();
    auto cy_ptr = geometry.getCylindersPtr();
    auto sq_ptr = geometry.getSquaresPtr();
    auto tr_ptr = geometry.getTrianglesPtr();

    auto _intersect = [&result, &ray](AGeomerty* geometry) {
        auto distance = geometry->intersect(ray);
        if (distance.has_value() && *distance < result.second)
            result = { geometry, *distance };
    };

    if (sp_ptr)
        for (size_t i = 0;  i < counter.getSpheresCount(); i++) {
            _intersect(sp_ptr);
            sp_ptr++;
        }

    if (pl_ptr)
        for (size_t i = 0; i < counter.getPlanesCount(); i++) {
            _intersect(pl_ptr);
            pl_ptr++;
        }

    if (cy_ptr)
        for (size_t i = 0; i < counter.getCylindersCount(); i++) {
            _intersect(cy_ptr);
            cy_ptr++;
        }

    if (sq_ptr)
        for (size_t i = 0; i < counter.getSquaresCount(); i++) {
            _intersect(sq_ptr);
            sq_ptr++;
        }

    if (tr_ptr)
        for (size_t i = 0; i < counter.getTrianglesCount(); i++) {
            _intersect(tr_ptr);
            tr_ptr++;
        }

    return result;
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
    double distanceToLight = Vec3d::vectorFromPoints(light.position, intersectionPoint).length();
    Ray ray(light.position, intersectionPoint);

    std::pair<AGeomerty *, double> closestShape = findNearestIntersect(ray);
    if (closestShape.second < distanceToLight && closestShape.first != intersectedShape.first) {
        return false;
    }
    return true;
}
