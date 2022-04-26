#include "Triangle.cuh"
#include "cudaDeviceDataPrint.cuh"
#include <stdio.h>

__device__ __host__ cuTriangle::cuTriangle() : cuAGeometry(cuColor{0.0,0.0,0.0})
{
	vertexes[0] = { 0.0,0.0,0.0 };
	vertexes[1] = { 0.0,0.0,0.0 };
	vertexes[2] = { 0.0,0.0,0.0 };
}

__device__ __host__ cuTriangle::cuTriangle(cuColor color_, cuVertex a, cuVertex b, cuVertex c) : cuAGeometry(cuColor{ 0.0,0.0,0.0 })
{
	vertexes[0] = a;
	vertexes[1] = b;
	vertexes[2] = c;
}

void cuTriangle::print()
{
    printf("Triangle:\n");
    printVector(color, "color: ");
    for (size_t i = 0; i < 3; i++)
    {
        printf("vertex #%i: ", i);
        printVector(vertexes[i], "");
    }
}

__device__ __host__ cuPair<double, bool> cuTriangle::intersect(const cuRay& ray)
{
    cuVec3d dir = cuVec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();
    cuVec3d e1 = vertexes[1] - vertexes[0];
    cuVec3d e2 = vertexes[2] - vertexes[0];

    cuVec3d pvec = cuCross(dir, e2);

    double det = cuDot(e1, pvec);

    if (det < 0.00001)
        return { 0.0, false };

    double inv_det = 1 / det;

    cuVec3d tvec = ray.Origin() - vertexes[0];
    double u = cuDot(tvec, pvec) * inv_det;
    if (u < 0 || u > 1)
        return { 0.0, false };

    cuVec3d qvec = cuCross(tvec, e1);
    double v = cuDot(dir, qvec) * inv_det;
    if (v < 0 || u + v > 1)
        return { 0.0, false };

    return { cuDot(e2, qvec) * inv_det, true };
}

__device__ __host__ cuVec3d cuTriangle::getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist)
{
    cuVec3d v1 = vertexes[1] - vertexes[0];
    cuVec3d v2 = vertexes[2] - vertexes[0];

    cuVec3d normal = cuCross(v1, v2).normalized();
    double a = cuDot(normal, view);
    if (acos(a) > (3.1415 / 2))
        return normal.inverse();
    return normal;
}
