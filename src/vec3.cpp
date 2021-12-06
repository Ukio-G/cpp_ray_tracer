#include "vec3.h"
#include <cmath>

vec3::vec3(): x(0), y(0), z(0){}

vec3::vec3(double x, double y, double z): x(x), y(y), z(z){}

vec3::~vec3() {}

vec3 vec3::operator+(const vec3& v) const {
	return vec3(x + v.x, y + v.y, z + v.z);
}

vec3 vec3::operator-(const vec3& v) const {
	return vec3(x - v.x, y - v.y, z - v.z);
}

vec3 vec3::operator*(double a) const {
	return vec3(x * a, y * a, z * a);
}

vec3 vec3::operator/(double a) const {
	return vec3(x / a, y / a, z / a);
}

vec3& vec3::operator=(const vec3& v) {
	if (this != &v) {
		x = v.x;
		y = v.y;
		z = v.z;
	}
	return *this;
}

double& vec3::operator[](int i) {
	if (i == 0)
		return x;
	if (i == 1)
		return y;
    return z;
}

std::ostream& operator<<(std::ostream&out, vec3& v) {
	out << v[0] << ' ' << v[1] << ' ' << v[2];
	return out;
}

double vec3::X() {
	return x;
}

double vec3::Y() {
	return y;
}

double vec3::Z() {
	return z;
}

double vec3::length() {
	return std::sqrt(x * x + y * y + z * z);
}
