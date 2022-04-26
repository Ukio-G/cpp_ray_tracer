#include "Cylinder.cuh"
#include <stdio.h>
#include "cudaDeviceDataPrint.cuh"

cuCylinder::cuCylinder() : cuAGeometry({ 0.0,0.0,0.0 }), diameter(0), height(0), position({ 0.0,0.0,0.0 }), direction({ 0.0,0.0,0.0 }), radius(0.0) { }

cuCylinder::cuCylinder(cuColor color_, double diameter_, double height_, cuVec3d position_, cuVec3d direction_)
	: cuAGeometry(color_), diameter(diameter_), height(height_), position(position_), direction(direction_.normalized()), radius(diameter_ / 2.0) {
}
void cuCylinder::print()
{
    printf("Cylinder: d:%f r:%f h:%f\n", diameter, radius, height);
    printVector(position, "position: ");
    printVector(direction, "direction: ");
}

cuPair<double, bool> cuCylinder::intersect(const cuRay& ray)
{
    cuVec3d coeff;

    cuVec3d  dir = cuVec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();

    auto temp1 = dir - (direction * cuDot(dir, direction));
    coeff[0] = cuDot(temp1, temp1);
    auto delta = ray.Origin() - position;
    auto temp2 = delta - (direction * cuDot(delta, direction));
    coeff[1] = 2 * cuDot(temp1, temp2);
    coeff[2] = cuDot(temp2, temp2) - (radius * radius);

    auto roots = solveSquareEq<double>(coeff);
    if (!roots.has_value())
        return { 0.0, false };

    /* Check candidates */
    auto dist_1 = checkCandidate(ray.Origin(), dir, (*roots).first);
    auto dist_2 = checkCandidate(ray.Origin(), dir, (*roots).second);
    if (!dist_1.second && !dist_2.second)
        return { 0.0, false };

    auto dist_1_ = (dist_1.second) ? dist_1.first : 10000000.0;
    auto dist_2_ = (dist_2.second) ? dist_2.first : 10000000.0;

    auto min_distance = cuMin(dist_1_, dist_2_);
    cuVec3d q = ray.Origin() + (dir * min_distance);
    return { (ray.Origin() - q).length(), true };
}

cuVec3d cuCylinder::getNormalInPoint(const cuVec3d& intersectionPoint, const cuVec3d& view, const cuRay& ray, double dist)
{
	return cuVec3d();
}

cuPair<double, bool> cuCylinder::checkCandidate(cuVec3d originRay, cuVec3d rayDirection, double dist)
{
    auto q = originRay + rayDirection * dist;

    if (dist >= 0 && cuDot(direction, q - _bottomPoint) > 0 && cuDot(direction, q - _topPoint) < 0)
        return { dist, false };
    return { 0.0, false };
}
