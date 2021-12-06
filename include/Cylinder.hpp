#ifndef CYLINDER_HPP
#define CYLINDER_HPP

#include "AGeomerty.hpp"

class Cylinder : public AGeomerty {
public:
    double diameter;
    double height;

    Vector<double, 3> position;
    Vector<double, 3> direction;
};

#endif
