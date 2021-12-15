#include "geometry/Plane.hpp"

Plane::Plane(Color color_, Vector<double, 3> position_, Vector<double, 3> normal_) : AGeomerty(color_), position(position_), normal(normal_) { }

Plane::Plane() : AGeomerty(), position({0.0, 0.0, 0.0}), normal({0.0, 0.0, 0.0}) { }

std::optional<double> Plane::intersect(const Ray &ray) {
    Vec3d viewDir = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction());
    double dotNormal = dot(normal, viewDir);
    if (dotNormal != 0.0) {
        double t = dot(Vec3d::vectorFromPoints(ray.Origin(), position), normal) / dotNormal;
        if (t < 0)
            return std::nullopt;
        auto pp = (ray.Origin() + viewDir * t);
        double dist = (pp - ray.Origin()).length();
        return dist;
    }
    return std::nullopt;
}

Vec3d Plane::getNormalInPoint(const Vec3d &intersectionPoint, const Vec3d &view) {
    if (acos(dot(normal, view)) > (M_PI / 2))
        return normal.inverse();
    return normal;
}
