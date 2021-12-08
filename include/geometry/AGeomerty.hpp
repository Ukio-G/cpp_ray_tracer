#ifndef AGEOMERTY_HPP
#define AGEOMERTY_HPP

#include "Vector.hpp"
#include "ray.h"

class AGeomerty {
public:
    AGeomerty();
    AGeomerty(Color & color_);
    Color color;
    virtual bool intersect(const Ray & ray) = 0;
};


#endif
