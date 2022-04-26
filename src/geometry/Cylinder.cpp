#include <math/Matrix.hpp>
#include <vector>
#include "math/math_utils.hpp"
#include "geometry/Cylinder.hpp"

Cylinder::Cylinder() : AGeomerty(), diameter(0), height(0), position({0.0,0.0,0.0}), direction({0.0,0.0,0.0}), radius(0.0) {}

Cylinder::Cylinder(Color color_, double diameter_, double height_, Vector<double, 3> position_, Vector<double, 3> direction_)
: AGeomerty(color_), diameter(diameter_), height(height_), position(position_), direction(direction_.normalized()), radius(diameter_ / 2.0) {
    auto invNormal = direction.inverse();
    _bottomPoint = position + (invNormal * (height / 2.0));
    _topPoint = _bottomPoint + (direction * height);
    initBoundBox();
}

std::optional<double> Cylinder::checkCandidate(Vec3d originRay, Vec3d rayDirection, double dist) {
    auto q = originRay + rayDirection * dist;

    if (dist >= 0 && dot(direction, q - _bottomPoint) > 0 && dot(direction, q - _topPoint) < 0)
        return dist;
    return std::nullopt;
}

std::vector<Sphere> Cylinder::initBoundBox() {
    std::vector<Sphere> result;

    double x_max;
    double x_min;

    double y_max;
    double y_min;

    double z_max;
    double z_min;

    // Diff angles
    auto cylinder_angles = getAngles( direction.normalized() );
    auto main_cylinder_angles = getAngles({0.0, 1.0, 0.0});
    auto rotate_angles = cylinder_angles - main_cylinder_angles;

    rotate_angles[0] = toRadian(rotate_angles[0]);
    rotate_angles[1] = toRadian(rotate_angles[1]);
    rotate_angles[2] = toRadian(rotate_angles[2]);


    // Rotation matrix
    auto rotate_matrix_x = Matrix3x3::rotateX(rotate_angles[0]);
    auto rotate_matrix_y = Matrix3x3::rotateY(rotate_angles[1]);
    auto rotate_matrix_z = Matrix3x3::rotateY(rotate_angles[2]);

    x_min = position[0] - radius;
    x_max = position[0] + radius;

    y_min = position[1] - height / 2;
    y_max = position[1] + height / 2;

    z_min = position[2] - radius;
    z_max = position[2] + radius;

    Vec3d Bbox[8];

    Bbox[0] = {x_max, y_max, z_min};
    Bbox[1] = BBoxMax =  {x_max, y_max, z_max};
    Bbox[2] = {x_min, y_max, z_max};
    Bbox[3] = {x_min, y_max, z_min};

    Bbox[4] = {x_max, y_min, z_min};
    Bbox[5] = {x_max, y_min, z_max};
    Bbox[6] = {x_min, y_min, z_max};
    Bbox[7] = BBoxMin = {x_min, y_min, z_min};

    for (int i = 0; i < 8; ++i) {
        Bbox[i] = (rotate_matrix_x * Bbox[i]);
        Bbox[i] = (rotate_matrix_y * Bbox[i]);
        Bbox[i] = (rotate_matrix_z * Bbox[i]);
        std::cout << "Bbox["<< i << "]: " << Bbox[i] << std::endl;

        Sphere sp(Color{255.0, 255.0, 255.0}, Bbox[i], 0.1);
        result.push_back(sp);
    }

    return result;
}

bool Cylinder::intersectBoundBox(const Ray &ray) {
    auto comp_wise_min = [](Vec3d & a, Vec3d & b) {
        Vec3d result;
        for (int i = 0; i < 3; ++i)
            result[i] = std::min(a[i], b[i]);
        return result;
    };

    auto comp_wise_max = [](Vec3d & a, Vec3d & b) {
        Vec3d result;
        for (int i = 0; i < 3; ++i)
            result[i] = std::max(a[i], b[i]);
        return result;
    };

    auto rayDir = ray.Direction();

    Vec3d  tMin = (BBoxMin - ray.Origin());
    tMin[0] = BBoxMin[0] / rayDir[0];
    tMin[1] = BBoxMin[1] / rayDir[1];
    tMin[2] = BBoxMin[2] / rayDir[2];

    Vec3d  tMax = (BBoxMax - ray.Origin());
    tMax[0] = BBoxMax[0] / rayDir[0];
    tMax[1] = BBoxMax[1] / rayDir[1];
    tMax[2] = BBoxMax[2] / rayDir[2];


    Vec3d t1 = comp_wise_min(tMin, tMax);
    Vec3d t2 = comp_wise_max(tMin, tMax);


    double tNear = std::max(std::max(t1[0], t1[1]), t1[2]);
    double tFar = std::min(std::min(t2[0], t2[1]), t2[2]);

    return tNear < tFar;
}


std::optional<double> Cylinder::intersect(const Ray &ray) {
    /*TODO: Check this */
    if (!intersectBoundBox(ray))
        return std::nullopt;

    Vec3d coeff;

    Vec3d dir = Vec3d::vectorFromPoints(ray.Origin(), ray.Direction()).normalized();

    auto temp1 = dir - (direction * dot(dir, direction));
    coeff[0] = dot(temp1, temp1);
    auto delta = ray.Origin() - position;
    auto temp2 = delta - (direction * dot(delta, direction));
    coeff[1] = 2 * dot(temp1, temp2);
    coeff[2] = dot(temp2, temp2) - (radius * radius);

    auto roots = solveSquareEq<double>(coeff);
    if (!roots.has_value())
        return std::nullopt;

    /* Check candidates */
    auto dist_1 = checkCandidate(ray.Origin(), dir, roots->first);
    auto dist_2 = checkCandidate(ray.Origin(), dir, roots->second);
    if (!dist_1 && !dist_2)
        return std::nullopt;
    auto min_distance = std::min(dist_1.value_or(1000000.0), dist_2.value_or(1000000.0));
    Vec3d q = ray.Origin() + (dir * min_distance);

    return (ray.Origin() - q).length();
}


Vec3d
Cylinder::getNormalInPoint(const Vec3d &intersection, const Vec3d &viewDir, const Ray &ray, double dist) {

    auto qc = (intersection - position).length();
    auto temp =  qc * qc - pow(radius, 2);

    Vec3d orig;

//    if (dot(cyl->normal, vec_from_points(cyl->origin, i->intersection_point)) > 0)

    if (dot(direction, Vec3d::vectorFromPoints(position, intersection)) > 0)
        orig = position + (direction * temp);
    else
        orig = position + (direction.inverse() * temp);

    auto n = Vec3d::vectorFromPoints(orig, intersection).normalized();

    if (Vec3d::vectorFromPoints(ray.Origin(), orig).length() > dist)
        return n;
    return n.inverse();
}
