#include <iostream>

#ifndef _VEC3_H
#define _VEC3_H

class vec3 {
protected:
	double x;
	double y;
	double z;
public:
	vec3();
	vec3(double x, double y, double z);
	~vec3();

	vec3 operator+(const vec3& v) const;
	vec3 operator-(const vec3& v) const;
	vec3 operator*(double a) const;
	vec3 operator/(double a) const;

	vec3& operator=(const vec3& v);
    friend std::ostream& operator<<(std::ostream& out, vec3& v);

    double& operator[](int i);

	double X();
	double Y();
	double Z();
	double length();


};
#endif // !_VEC3_H

