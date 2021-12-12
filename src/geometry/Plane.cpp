#include "geometry/Plane.hpp"

Plane::Plane(Color color_, Vector<double, 3> position_, Vector<double, 3> normal_) : AGeomerty(color_), position(position_), normal(normal_) { }

Plane::Plane() : AGeomerty(), position({0.0, 0.0, 0.0}), normal({0.0, 0.0, 0.0}) { }

bool Plane::intersect(const Ray &ray) {
    return false;
}
