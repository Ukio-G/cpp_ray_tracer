#include "ray.h"

Ray::Ray(){}

Ray::Ray(const vec3& O, const vec3& D) : origin(O), direction(D) {}

Ray::~Ray(){}

vec3 Ray::Origin() const {
	return origin;
}

vec3 Ray::Direction() const {
	return direction;
}

vec3 Ray::pointAt(double t) const {
	return origin + direction * t;
}