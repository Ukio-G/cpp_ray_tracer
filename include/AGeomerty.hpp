#ifndef AGEOMERTY_HPP
#define AGEOMERTY_HPP

#include "Vector.hpp"
#include "ray.h"

class AGeomerty {
public:
    AGeomerty();
    AGeomerty(const Vector<unsigned char, 3> & color);
    Vector<unsigned char, 3> color;
    virtual bool intersect(const Ray & ray) = 0;
};


#endif
