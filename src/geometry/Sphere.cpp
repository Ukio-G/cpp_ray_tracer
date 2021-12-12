#include "geometry/Sphere.hpp"

Sphere::Sphere(Color color_, Vector<double, 3> position_, double diameter_) : AGeomerty(color_), position(position_), diameter(diameter_) { }

Sphere::Sphere() : AGeomerty(), position({0.0, 0.0, 0.0}), diameter(0.0) { }

std::optional<double> Sphere::intersect(const Ray &ray) {
    return std::nullopt;
}

std::optional<Vec3d> Sphere::intersectPoint(const Ray &ray) {
    return std::nullopt;
}

Vec3d Sphere::getNormalInPoint(const Vec3d &intersectionPoint) {
    return Vec3d();
}
