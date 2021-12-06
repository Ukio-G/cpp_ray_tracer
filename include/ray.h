#include "vec3.h"
#ifndef _RAY_H
#define _RAY_H

class Ray
{
public:
	Ray();
	Ray(const vec3& O, const vec3& D);
	~Ray();

	vec3 Origin() const;
	vec3 Direction() const;
	vec3 pointAt(double t) const;

private:
	vec3 origin;
	vec3 direction;
};
#endif // !_RAY_H

