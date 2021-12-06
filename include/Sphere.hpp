#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "AGeomerty.hpp"
#include "Vector.hpp"

class Sphere : public AGeomerty {
public:
    Vector<double, 3> position;
    double diameter;
};


#endif
