#include "ray.h"

Ray::Ray(){}

Ray::Ray(const Vec3d& O, const Vec3d& D) : origin(O), direction(D) {}

Ray::~Ray(){}

Vec3d Ray::Origin() const {
	return origin;
}

Vec3d Ray::Direction() const {
	return direction;
}

Vec3d Ray::pointAt(double t) {
    //return {1.0,1.0,1.0};
	return origin + direction * t;
}