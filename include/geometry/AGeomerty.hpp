#ifndef AGEOMERTY_HPP
#define AGEOMERTY_HPP

#include <optional>
#include <any>
#include "Vector.hpp"
#include "ray.h"

class AGeomerty {
public:
    AGeomerty();
    AGeomerty(Color & color_);
    Color color;

    virtual std::optional<double> intersect(const Ray & ray) = 0;
    virtual Vec3d getNormalInPoint(const Vec3d &point, const Vec3d &view, const Ray &ray, double dist) = 0;
};


#endif
