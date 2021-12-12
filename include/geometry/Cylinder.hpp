#ifndef CYLINDER_HPP
#define CYLINDER_HPP

#include "AGeomerty.hpp"

class Cylinder : public AGeomerty {
public:
    Cylinder();
    Cylinder(Color color_, double diameter_, double height_, Vector<double, 3> position_, Vector<double, 3> direction_);

    double diameter;
    double height;

    Vector<double, 3> position;
    Vector<double, 3> direction;

    bool intersect(const Ray & ray);
};

inline std::ostream & operator<<(std::ostream &ostream, Cylinder & cylinder) {
    ostream << "color: " << cylinder.color;
    ostream << ", position: " << cylinder.position;
    ostream << ", diameter: " << cylinder.diameter;
    ostream << ", height: " << cylinder.height;
    ostream << ", direction: " << cylinder.direction;
    return ostream;
}

#endif
