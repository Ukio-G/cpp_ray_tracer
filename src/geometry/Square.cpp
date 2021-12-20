#include "math/math_utils.hpp"
#include "math/Matrix.hpp"
#include "geometry/Square.hpp"

Square::Square(Color color_, Vector<double, 3> center_, Vector<double, 3> direction_, double sizeSide_) :
AGeomerty(color_), center(center_), direction(direction_), sizeSide(sizeSide_) {
    initVertexes();
}

Square::Square() : AGeomerty(), center({0.0, 0.0, 0.0}), direction({0.0, 0.0, 0.0}), sizeSide(0.0) {

}

void Square::initVertexes() {
    double s = sizeSide / 2;
    vertexes[0] = Vec3d {s, s, 0.0};
    vertexes[1] = Vec3d {-s, s, 0.0};
    vertexes[2] = Vec3d {-s, -s, 0.0};
    vertexes[3] = Vec3d {s, -s, 0.0};

    Vec3d normalAngles = getAngles(direction);
    Vec3d startAngles = getAngles({0.0, 0.0, 1.0});
    Vec3d resultAngles = normalAngles - startAngles;
    resultAngles = {toRadian(resultAngles[0]), toRadian(resultAngles[1]), toRadian(resultAngles[2])};

    /* Rotate vertexes */
    Matrix3x3	rotate_matrix;

    rotate_matrix = Matrix3x3::rotateX(resultAngles[0]);
    for (int i = 0; i < 4; ++i)
        vertexes[i] = rotate_matrix * vertexes[i];

    rotate_matrix = Matrix3x3::rotateY(resultAngles[0]);
    for (int i = 0; i < 4; ++i)
        vertexes[i] = rotate_matrix * vertexes[i];

    /* Translate vertexes */
    for (int i = 0; i < 4; ++i)
        vertexes[i] = vertexes[i] + center;
}

std::optional<double> Square::intersect(const Ray &ray) {
    double t;
    Vec3d  dir;

    dir = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();
    t = dot(vertexes[0] - ray.Origin(), direction) / dot(dir, direction);

    if (t < 0)
        return std::nullopt;

    Vec3d m = ray.Origin() + dir * t;
    t = (m - ray.Origin()).length();

    auto temp = m - vertexes[0];
    auto e1 = dot(temp, vertexes[1] - vertexes[0]) / sizeSide;
    auto e2 = dot(temp, vertexes[3] - vertexes[0]) / sizeSide;

    if ((e1 > 0 && e1 < sizeSide) && (e2 > 0 && e2 < sizeSide))
        return t;

    return std::nullopt;
}

Vec3d Square::getNormalInPoint(const Vec3d &intersectionPoint, const Vec3d &view, const Ray &ray, double dist) {
    auto normal = direction.normalized();
    if (acos(dot(normal, view)) > (M_PI / 2))
        normal = normal.inverse();
    return (normal);
}
