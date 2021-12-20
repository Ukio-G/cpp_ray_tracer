#include "geometry/Sphere.hpp"
#include "math/math_utils.hpp"

Sphere::Sphere(Color color_, Vector<double, 3> position_, double diameter_) : AGeomerty(color_), position(position_), diameter(diameter_) { }

Sphere::Sphere() : AGeomerty(), position({0.0, 0.0, 0.0}), diameter(0.0) { }

std::optional<double> Sphere::intersect(const Ray &ray) {
    double      tc[2];
    double radius = diameter / 2.0;
    Vec3d l = position - ray.Origin();

    Vec3d ray_normalized = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();

    tc[0] = dot(l, ray_normalized);
    double d2 = dot(l, l) - tc[0] * tc[0];

    if (d2 > radius * radius)
        return std::nullopt;

    tc[1] = sqrt(radius * radius - d2);
    double d = tc[0] - tc[1];
    double t1 = tc[0] + tc[1];

    if (d < 0) {
        m_inverse = true;
        d = t1;
    }

    if (d < 0)
        return std::nullopt;
    return d;
}

Vec3d Sphere::getNormalInPoint(const Vec3d &intersectionPoint, const Vec3d &view, const Ray &ray, double dist) {
    Vec3d normal = Vec3d::vectorFromPoints(position, intersectionPoint);
    if (m_inverse)
        normal = normal.inverse();
    return normal.normalized();
}
