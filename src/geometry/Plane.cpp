#include "geometry/Plane.hpp"

Plane::Plane(Color color_, Vector<double, 3> position_, Vector<double, 3> normal_) : AGeomerty(color_), position(position_), normal(normal_) { }

Plane::Plane() : AGeomerty(), position({0.0, 0.0, 0.0}), normal({0.0, 0.0, 0.0}) { }

std::optional<double> Plane::intersect(const Ray &ray) {
    return std::nullopt;
}

std::optional<Vec3d> Plane::intersectPoint(const Ray &ray) {
    return std::nullopt;
}

Vec3d Plane::getNormalInPoint(const Vec3d &intersectionPoint) {
    return Vec3d();
}
