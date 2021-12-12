#include "Camera.hpp"

Camera::Camera(Vec3d position_, Vec3d direction_, double fov_) : position(position_), direction(direction_), fov(fov_) { }

Camera::Camera() : position(), direction(), fov(0.0) { }
