#ifndef SQUARE_HPP
#define SQUARE_HPP

#include "AGeomerty.hpp"

class Square : public AGeomerty {
public:
    Square();
    Square(Color color_, Vec3d center_, Vec3d direction_, double sizeSide_);
    Vec3d center;
    Vec3d direction;
    double sizeSide;
    std::optional<double> intersect(const Ray & ray);
    std::optional<Vec3d> intersectPoint(const Ray &ray);
    Vec3d getNormalInPoint(const Vec3d & intersectionPoint);
};

inline std::ostream & operator<<(std::ostream &ostream, Square & square) {
    ostream << "color: " << square.color;
    ostream << ", center: " << square.center;
    ostream << ", direction: " << square.direction;
    ostream << ", sizeSide: " << square.sizeSide;
    return ostream;
}

#endif
