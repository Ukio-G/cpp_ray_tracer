#ifndef AMBIENTLIGHT_HPP
#define AMBIENTLIGHT_HPP

#include "ALight.hpp"
#include "Vector.hpp"

class AmbientLight : public ALight {
public:
    AmbientLight() : ALight() {}
    AmbientLight(Color color, double ratio) : ALight(color, ratio) {}
};

inline std::ostream & operator<<(std::ostream &ostream, AmbientLight & ambientLight) {
    ostream << "color: " << ambientLight.color << ", ratio: " << ambientLight.ratio;
    return ostream;
}

#endif //AMBIENTLIGHT_HPP
