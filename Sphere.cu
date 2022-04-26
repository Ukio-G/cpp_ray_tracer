#include "Sphere.cuh"
#include "AGeometry.cuh"
#include <stdio.h>
#include "cudaDeviceDataPrint.cuh"

__host__ __device__ cuSphere::cuSphere(cuColor color_, cuVec3d position_, double diameter_) : cuAGeometry(color_), position(position_), diameter(diameter_), radius(diameter_ / 2.0), m_radius_sq(radius * radius) { }

__host__ __device__ cuSphere::cuSphere() : cuAGeometry(cuColor{ 0.0, 0.0, 0.0 }), position({ 0.0, 0.0, 0.0 }), diameter(0.0), radius(0.0), m_radius_sq(0.0) { }

__host__ __device__ static void printVector(const cuVec3d& vec) {
    printf("(%f, %f, %f)\n", vec[0], vec[1], vec[2]);
}

__host__ __device__ static void printRay(const cuRay& ray) {
    printf("Origin: ( %f, %f, %f ), Direction: ( %f, %f, %f ) \n", ray.Origin()[0], ray.Origin()[1], ray.Origin()[2], ray.Direction()[0], ray.Direction()[1], ray.Direction()[2]);
}
__host__ __device__ static void printSphere(const cuSphere& sp) {
    printf("Position sphere: ");
    printVector(sp.position);

    printf("\nColor sphere: ");
    printVector(sp.color);

    printf("\nDiameter sphere: %f, Radius sphere: %f, Radius^2 sphere: %f\n", sp.diameter, sp.radius, sp.m_radius_sq);
}

__host__ __device__ cuPair<double, bool> cuSphere::intersect(const cuRay& ray) {

#ifdef NEW_SPHERE_ALGO
    auto o = ray.Origin();
    auto d = ray.Direction();

    auto oc = position - o;
    double t = cuDot(oc, d);

    auto q = o + d * t;
    auto b_sq = (q - position).length();
    b_sq = b_sq * b_sq;

    if (b_sq > m_radius_sq)
        return { 0.0, false };
    auto a = sqrt(m_radius_sq - b_sq);

    auto q1 = (o + d * (t + a)).length();
    auto q2 = (o + d * (t - a)).length();

    return { (q1 < q2) ? q1 : q2, true };


#else
    double tc[2];
    cuVec3d l = position - ray.Origin();

    cuVec3d ray_normalized = cuVec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();

    tc[0] = cuDot(l, ray_normalized);
    double d2 = cuDot(l, l) - tc[0] * tc[0];

    if (d2 > m_radius_sq)
        return { 100000.0, false };

    tc[1] = sqrt(m_radius_sq - d2);
    double d = tc[0] - tc[1];
    double t1 = tc[0] + tc[1];

    bool inv = false;
    if (d < 0.0) {
        inv = true;
        d = t1;
    }

    if (d < 0.0)
        return { 100000.0, false };

    if (m_inversable && inv)
        m_inverse = true;
    m_inversable = false;
    return { d, true };
#endif
}

cuVec3d cuSphere::getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist) {
    cuVec3d normal = cuVec3d::vectorFromPoints(position, intersectionPoint);
    if (m_inverse)
        normal = normal.inverse();
    return normal.normalized();
}

void cuSphere::print()
{
    printVector(position, "sphere position");
    printVector(color, "Color sphere");
    printf("Diameter sphere: %f, Radius sphere: %f, Radius^2 sphere: %f\n", diameter, radius, m_radius_sq);
}
