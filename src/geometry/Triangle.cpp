#include "geometry/Triangle.hpp"

Triangle::Triangle() : AGeomerty() {
    vertexes[0] = {0.0, 0.0, 0.0};
    vertexes[1] = {0.0, 0.0, 0.0};
    vertexes[2] = {0.0, 0.0, 0.0};
}

Triangle::Triangle(Color color_, Vertex a, Vertex b, Vertex c) : AGeomerty(color_) {
    vertexes[0] = a;
    vertexes[1] = b;
    vertexes[2] = c;
}

std::optional<double> Triangle::intersect(const Ray &ray) {
    return std::nullopt;
}

std::optional<Vec3d> Triangle::intersectPoint(const Ray &ray) {
    return std::nullopt;
}

Vec3d Triangle::getNormalInPoint(const Vec3d &intersectionPoint) {
    return Vec3d();
}
