#ifndef LIGHTSOURCE_HPP
#define LIGHTSOURCE_HPP

#include "ALight.hpp"

class LightSource : public ALight {
public:
    LightSource() : ALight(), position() {}
    LightSource(Color color, double ratio, Vec3d position) : ALight(color, ratio), position(position) {}

    Vec3d position;
};

inline std::ostream & operator<<(std::ostream &ostream, LightSource & lightSource) {
    ostream << "color: " << lightSource.color;
    ostream << ", ratio: " << lightSource.ratio;
    ostream << ", position: " << lightSource.position;
    return ostream;
}

#endif //LIGHTSOURCE_HPP
