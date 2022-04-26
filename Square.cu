#include "Square.cuh"
#include <stdio.h>
#include "cudaDeviceDataPrint.cuh"

__host__ __device__ cuSquare::cuSquare() : cuAGeometry({ 0.0, 0.0, 0.0 }), center({ 0.0, 0.0, 0.0 }), direction({ 0.0, 0.0, 0.0 }), sizeSide(0.0)
{
}

__host__ __device__ cuSquare::cuSquare(cuColor color_, cuVec3d center_, cuVec3d direction_, double sizeSide_) :
	cuAGeometry(color_), center(center_), direction(direction_), sizeSide(sizeSide_) 
{
}

void cuSquare::print()
{
    printf("Square: sizeSide:%f \n", sizeSide);
    printVector(color, "color: ");
    printVector(center, "center: ");
    printVector(direction, "direction: ");

    for (size_t i = 0; i < 4; i++)
    {
        printf("vertex #%i: ", i);
        printVector(vertexes[i], "");
    }
}

__host__ __device__ cuPair<double, bool> cuSquare::intersect(const cuRay& ray)
{
    double t;
    cuVec3d  dir;

    dir = cuVec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();
    t = cuDot(vertexes[0] - ray.Origin(), direction) / cuDot(dir, direction);

    if (t < 0)
        return { 0.0, false };

    cuVec3d m = ray.Origin() + dir * t;
    t = (m - ray.Origin()).length();

    auto temp = m - vertexes[0];
    auto e1 = cuDot(temp, vertexes[1] - vertexes[0]) / sizeSide;
    auto e2 = cuDot(temp, vertexes[3] - vertexes[0]) / sizeSide;

    if ((e1 > 0 && e1 < sizeSide) && (e2 > 0 && e2 < sizeSide))
        return { t, true };

    return { 0.0, false };
}

__host__ __device__ cuVec3d cuSquare::getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist)
{
    auto normal = direction.normalized();
    if (acos(cuDot(normal, view)) > (3.1415 / 2))
        normal = normal.inverse();
    return (normal);
}

__host__ __device__ void cuSquare::initVertexes()
{
    double s = sizeSide / 2;
    vertexes[0] = cuVec3d{ s, s, 0.0 };
    vertexes[1] = cuVec3d{ -s, s, 0.0 };
    vertexes[2] = cuVec3d{ -s, -s, 0.0 };
    vertexes[3] = cuVec3d{ s, -s, 0.0 };

    cuVec3d normalAngles = cuGetAngles(direction);
    cuVec3d startAngles = cuGetAngles({ 0.0, 0.0, 1.0 });
    cuVec3d resultAngles = normalAngles - startAngles;
    resultAngles = { cuToRadian(resultAngles[0]), cuToRadian(resultAngles[1]), cuToRadian(resultAngles[2]) };

    /* Rotate vertexes */
    cuMatrix3x3	rotate_matrix;

    rotate_matrix = cuMatrix3x3::rotateX(resultAngles[0]);
    for (int i = 0; i < 4; ++i)
        vertexes[i] = rotate_matrix * vertexes[i];

    rotate_matrix = cuMatrix3x3::rotateY(resultAngles[0]);
    for (int i = 0; i < 4; ++i)
        vertexes[i] = rotate_matrix * vertexes[i];

    /* Translate vertexes */
    for (int i = 0; i < 4; ++i)
        vertexes[i] = vertexes[i] + center;
}
