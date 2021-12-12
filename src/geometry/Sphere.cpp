#include "geometry/Sphere.hpp"

Sphere::Sphere(Color color_, Vector<double, 3> position_, double diameter_) : AGeomerty(color_), position(position_), diameter(diameter_) { }

Sphere::Sphere() : AGeomerty(), position({0.0, 0.0, 0.0}), diameter(0.0) { }

bool Sphere::intersect(const Ray &ray) {
    return false;
}
