#include "math/math_utils.hpp"
#include "geometry/Cylinder.hpp"

Cylinder::Cylinder() : AGeomerty(), diameter(0), height(0), position({0.0,0.0,0.0}), direction({0.0,0.0,0.0}), radius(0.0) {}

Cylinder::Cylinder(Color color_, double diameter_, double height_, Vector<double, 3> position_, Vector<double, 3> direction_)
: AGeomerty(color_), diameter(diameter_), height(height_), position(position_), direction(direction_), radius(diameter_ / 2.0) {
    auto invNormal = direction.inverse();
    _bottomPoint = position + (invNormal * (height / 2.0));
    _topPoint = _bottomPoint + (direction * height);
}

std::optional<double> Cylinder::checkCandidate(Vec3d originRay, Vec3d rayDirection, double dist) {
    auto q = originRay + rayDirection * dist;

    if (dist >= 0 && dot(direction, q - _bottomPoint) > 0 && dot(direction, q - _topPoint) < 0)
        return dist;
    return std::nullopt;
}

std::optional<double> Cylinder::intersect(const Ray &ray) {
    Vec3d coeff;

    Vec3d dir = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();


    auto temp1 = dir - (direction * dot(dir, direction));
    coeff[0] = dot(temp1, temp1);
    auto delta = ray.Origin() - position;
    auto temp2 = delta - (direction * dot(delta, direction));
    coeff[1] = 2 * dot(temp1, temp2);
    coeff[2] = dot(temp2, temp2) - (radius * radius);

    auto roots = solveSquareEq<double>(coeff);
    if (!roots.has_value())
        return std::nullopt;

    /* Check candidates */
    auto dist_1 = checkCandidate(ray.Origin(), dir, roots->first);
    auto dist_2 = checkCandidate(ray.Origin(), dir, roots->second);
    if (!dist_1 && !dist_2)
        return std::nullopt;
    auto min_distance = std::min(dist_1.value_or(1000000.0), dist_2.value_or(1000000.0));
    Vec3d q = ray.Origin() + (dir * min_distance);

    return (ray.Origin() - q).length();
}


Vec3d
Cylinder::getNormalInPoint(const Vec3d &intersection, const Vec3d &viewDir, const Ray &ray, double dist) {

    auto qc = (intersection - position).length();
    auto temp =  qc * qc - pow(radius, 2);

    Vec3d orig;

    if (dot(direction, Vec3d::vectorFromPoints(position, intersection)) > 0)
        orig = position + (direction * temp);
    else
        orig = position + (direction.inverse() * temp);

    auto n = Vec3d::vectorFromPoints(orig, intersection).normalized();

    if (Vec3d::vectorFromPoints(ray.Origin(), orig).length() > dist)
        return n;
    return n.inverse();
}
