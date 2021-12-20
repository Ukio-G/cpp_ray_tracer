#ifndef PLANE_HPP
#define PLANE_HPP

#include "AGeomerty.hpp"

class Plane : public AGeomerty {
public:
    Plane();
    Plane(Color color_, Vector<double, 3> position_, Vector<double, 3> normal_);

    Vector<double, 3> position;
    Vector<double, 3> normal;

    std::optional<double> intersect(const Ray & ray);
    Vec3d getNormalInPoint(const Vec3d &intersectionPoint, const Vec3d &view, const Ray &ray, double dist);
};

inline std::ostream & operator<<(std::ostream &ostream, Plane & plane) {
    ostream << "color: " << plane.color;
    ostream << ", position: " << plane.position;
    ostream << ", normal: " << plane.normal;
    return ostream;
}
#endif //PLANE_HPP
