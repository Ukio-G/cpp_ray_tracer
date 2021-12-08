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
};

#endif
