#ifndef ALIGHT_HPP
#define ALIGHT_HPP

#include "Vector.hpp"

class ALight {
public:
    ALight() : color(), ratio(0.0) { }
    ALight(Color color, double ratio) : color(color), ratio(ratio) { }
    virtual ~ALight() {};
    Color color;
    double ratio;
};

#endif //ALIGHT_HPP
