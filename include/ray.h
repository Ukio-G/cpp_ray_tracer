#ifndef _RAY_H
#define _RAY_H

#include "Vector.hpp"

class Ray
{
public:
	Ray();
	Ray(const Vec3d& O, const Vec3d& D);
	~Ray();

    Vec3d Origin() const;
    Vec3d Direction() const;
    Vec3d pointAt(double t);

private:
    Vec3d origin;
    Vec3d direction;
};
#endif // !_RAY_H

