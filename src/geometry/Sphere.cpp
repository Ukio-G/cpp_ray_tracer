#include "geometry/Sphere.hpp"
#include "math/math_utils.hpp"

Sphere::Sphere(Color color_, Vector<double, 3> position_, double diameter_) : AGeomerty(color_), position(position_), diameter(diameter_), radius(diameter_ / 2), m_radius_sq(radius * radius) { }

Sphere::Sphere() : AGeomerty(), position({0.0, 0.0, 0.0}), diameter(0.0), radius(0.0), m_radius_sq(0.0) { }

std::optional<double> Sphere::intersect(const Ray &ray) {

#ifdef NEW_SPHERE_ALGO
    auto o = ray.Origin();
    auto d = ray.Direction();

    auto oc = position - o;
    double t = dot(oc, d);

    auto q = o + d * t;
    auto b_sq = (q - position).length();
    b_sq = b_sq * b_sq;

    if (b_sq > m_radius_sq)
        return std::nullopt;
    auto a = sqrt(m_radius_sq - b_sq);

    auto q1 = (o + d * (t + a)).length();
    auto q2 = (o + d * (t - a)).length();

    return (q1 < q2) ? q1 : q2;


#else
    double tc[2];
    Vec3d l = position - ray.Origin();

    Vec3d ray_normalized = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();

    tc[0] = dot(l, ray_normalized);
    double d2 = dot(l, l) - tc[0] * tc[0];

    if (d2 > m_radius_sq)
        return std::nullopt;

    tc[1] = sqrt(m_radius_sq - d2);
    double d = tc[0] - tc[1];
    double t1 = tc[0] + tc[1];

    bool inv = false;
    if (d < 0) {
        inv = true;
        d = t1;
    }

    if (d < 0)
        return std::nullopt;

    if (m_inversable && inv)
        m_inverse = true;
    m_inversable = false;
    return d;
#endif
}

Vec3d Sphere::getNormalInPoint(const Vec3d &intersectionPoint, const Vec3d &view, const Ray &ray, double dist) {
    Vec3d normal = Vec3d::vectorFromPoints(position, intersectionPoint);
    if (m_inverse)
        normal = normal.inverse();
    return normal.normalized();
}
