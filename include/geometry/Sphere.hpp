#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "AGeomerty.hpp"
#include "Vector.hpp"

class Sphere : public AGeomerty {
public:
    Sphere();
    Sphere(Color color_, Vector<double, 3> position_, double diameter_);
    Vector<double, 3> position;
    double diameter;
};


#endif
