#include "geometry/Triangle.hpp"
#include "math/math_utils.hpp"
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

/* Алгоритм Моллера — Трумбора */
std::optional<double> Triangle::intersect(const Ray &ray) {
    auto dir = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();
    auto e1 = vertexes[1] - vertexes[0];
    auto e2 = vertexes[2] - vertexes[0];
    auto pvec = cross(dir, e2);
    auto det = dot(e1, pvec);

    if (det > -0.00001 && det < 0.00001)
        return std::nullopt;
    auto inv_det = 1.0 / det;

    auto tvec = ray.Origin() - vertexes[0];

    auto u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return std::nullopt;

    auto qvec = cross(tvec, e1);
    auto v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return std::nullopt;
    auto t = dot(e2, qvec) * inv_det;
    return t;
}

Vec3d
Triangle::getNormalInPoint(const Vec3d &intersectionPoint, const Vec3d &view, const Ray &ray, double dist) {
    Vec3d v1 = vertexes[1] - vertexes[0];
    Vec3d v2 = vertexes[2] - vertexes[0];

    Vec3d normal = cross(v1, v2).normalized();
    double a = dot(normal, view);
    if (acos(a) > (M_PI / 2))
        return normal.inverse();
    return normal;
}
