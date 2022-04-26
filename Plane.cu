#include "Plane.cuh"
#include <stdio.h>
#include "cudaDeviceDataPrint.cuh"

__host__ __device__  cuPlane::cuPlane() : cuAGeometry({ 0.0, 0.0, 0.0 }), position({ 0.0, 0.0, 0.0 }), normal({ 0.0, 0.0, 0.0 }) {
}

__host__ __device__  cuPlane::cuPlane(cuColor color_, cuVec3d position_, cuVec3d normal_) : cuAGeometry(color_), position(position_), normal(normal_) {
}

void cuPlane::print()
{
    printf("Plane:\n");
    printVector(position, "position: ");
    printVector(normal, "direction: ");
    printVector(color, "color: ");
}

__host__ __device__  cuPair<double, bool> cuPlane::intersect(const cuRay& ray)
{
    cuVec3d viewDir = cuVec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();
    double cuDotNormal = cuDot(normal, viewDir);
    if (cuDotNormal != 0.0) {
        double t = cuDot(cuVec3d::vectorFromPoints(ray.Origin(), position), normal) / cuDotNormal;
        if (t < 0)
            return { 0.0, false };
        auto pp = (ray.Origin() + viewDir * t);
        double dist = (pp - ray.Origin()).length();
        return { dist, true };
    }
    return { 0.0, false };
}

__host__ __device__  cuVec3d cuPlane::getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist)
{
    if (acos(cuDot(normal, view)) > (3.1415 / 2))
        return normal.inverse();
    return normal;
}
