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
    bool intersect(const Ray & ray);
};

inline std::ostream & operator<<(std::ostream &ostream, Square & square) {
    ostream << "color: " << square.color;
    ostream << ", center: " << square.center;
    ostream << ", direction: " << square.direction;
    ostream << ", sizeSide: " << square.sizeSide;
    return ostream;
}

#endif
