#include "geometry/Cylinder.hpp"

Cylinder::Cylinder() : AGeomerty(), diameter(0), height(0), position({0.0,0.0,0.0}), direction({0.0,0.0,0.0}) {}

Cylinder::Cylinder(Color color_, double diameter_, double height_, Vector<double, 3> position_, Vector<double, 3> direction_)
: AGeomerty(color_), diameter(diameter_), height(height_), position(position_), direction(direction_) { }

bool Cylinder::intersect(const Ray &ray) {
    return false;
}
