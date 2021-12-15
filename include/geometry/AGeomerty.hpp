#ifndef AGEOMERTY_HPP
#define AGEOMERTY_HPP

#include <optional>
#include "Vector.hpp"
#include "ray.h"

class AGeomerty {
public:
    AGeomerty();
    AGeomerty(Color & color_);
    Color color;

    virtual std::optional<double> intersect(const Ray & ray) = 0;
    virtual Vec3d getNormalInPoint(const Vec3d &point, const Vec3d &view) = 0;
};


#endif
