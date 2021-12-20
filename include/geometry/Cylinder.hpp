#ifndef CYLINDER_HPP
#define CYLINDER_HPP

#include "AGeomerty.hpp"

class Cylinder : public AGeomerty {
public:
    Cylinder();
    Cylinder(Color color_, double diameter_, double height_, Vector<double, 3> position_, Vector<double, 3> direction_);

    double diameter;
    double radius;
    double height;

    Vector<double, 3> position;
    Vector<double, 3> direction;

    std::optional<double> intersect(const Ray & ray);
    Vec3d getNormalInPoint(const Vec3d &intersectionPoint, const Vec3d &view, const Ray &ray, double dist);

private:
    Vec3d _bottomPoint;
    Vec3d _topPoint;

    std::optional<double> checkCandidate(Vec3d originRay, Vec3d rayDirection, double dist);
};

inline std::ostream & operator<<(std::ostream &ostream, Cylinder & cylinder) {
    ostream << "color: " << cylinder.color;
    ostream << ", position: " << cylinder.position;
    ostream << ", diameter: " << cylinder.diameter;
    ostream << ", height: " << cylinder.height;
    ostream << ", direction: " << cylinder.direction;
    return ostream;
}

#endif
