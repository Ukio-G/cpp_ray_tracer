#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "Vector.hpp"

class Camera {
public:
    Camera();
    Camera(Vec3d position_, Vec3d direction_, double fov_);

    Vec3d position;
    Vec3d direction;
    double fov;
};

inline std::ostream & operator<<(std::ostream &ostream, Camera & camera) {
    ostream << "position: " << camera.position;
    ostream << ", direction: " << camera.direction;
    ostream << ", fov: " << camera.fov;
    return ostream;
}

#endif
