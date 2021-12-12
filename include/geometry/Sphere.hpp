#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "AGeomerty.hpp"
#include "Vector.hpp"

class Sphere : public AGeomerty {
public:
    Sphere();
    Sphere(Color color_, Vec3d position_, double diameter_);
    Vec3d position;
    double diameter;
    std::optional<double> intersect(const Ray & ray);
    std::optional<Vec3d> intersectPoint(const Ray & ray);
    Vec3d getNormalInPoint(const Vec3d & intersectionPoint);
};

inline std::ostream & operator<<(std::ostream &ostream, Sphere & sphere) {
    ostream << "color: " << sphere.color;
    ostream << ", position: " << sphere.position;
    ostream << ", diameter: " << sphere.diameter;
    return ostream;
}

#endif
