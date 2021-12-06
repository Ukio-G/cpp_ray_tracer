#ifndef PLANE_HPP
#define PLANE_HPP

#include "AGeomerty.hpp"

class Plane : public AGeomerty {
public:
    Vector<double, 3> position;
    Vector<double, 3> direction;
};


#endif //PLANE_HPP
