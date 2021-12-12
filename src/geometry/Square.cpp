#include "geometry/Square.hpp"

Square::Square(Color color_, Vector<double, 3> center_, Vector<double, 3> direction_, double sizeSide_) :
AGeomerty(color_), center(center_), direction(direction_), sizeSide(sizeSide_) { }

Square::Square() : AGeomerty(), center({0.0, 0.0, 0.0}), direction({0.0, 0.0, 0.0}), sizeSide(0.0) { }

std::optional<double> Square::intersect(const Ray &ray) {
    return std::nullopt;
}

std::optional<Vec3d> Square::intersectPoint(const Ray &ray) {
    return std::nullopt;
}

Vec3d Square::getNormalInPoint(const Vec3d &intersectionPoint) {
    return Vec3d();
}
