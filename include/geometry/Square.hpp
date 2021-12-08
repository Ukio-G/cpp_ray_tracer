#ifndef SQUARE_HPP
#define SQUARE_HPP

#include "AGeomerty.hpp"

class Square : public AGeomerty {
public:
    Square();
    Square(Color color_, Vector<double, 3> center_, Vector<double, 3> direction_, double sizeSide_);
    Vector<double, 3> center;
    Vector<double, 3> direction;
    double sizeSide;
};

#endif
