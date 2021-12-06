#ifndef SQUARE_HPP
#define SQUARE_HPP

#include "AGeomerty.hpp"

class Square : public AGeomerty {
public:
    Vector<double, 3> center;
    Vector<double, 3> direction;
    double sizeSide;
};

#endif
