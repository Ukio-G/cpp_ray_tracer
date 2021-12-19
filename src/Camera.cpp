#include <ray.h>
#include <math/math_utils.hpp>
#include <FrameBuffer.hpp>
#include "Camera.hpp"
#include "math/Matrix.hpp"

Camera::Camera(Vec3d position_, Vec3d direction_, double fov_) : position(position_), direction(direction_), fov(fov_) { }

Camera::Camera() : position(), direction(), fov(0.0) { }


template<class T, class U>
int f(T t, U u, int i) {

}


Ray Camera::computeRayForPixel(unsigned int x, unsigned int y, FrameBuffer & framebuffer) {
    auto start = position;
    auto half_width = (double)framebuffer.width / 2.0;

    auto x_r = (double)x - half_width;
    auto y_r = (double)framebuffer.height / 2.0 - (double)y;
    auto z_r = half_width / tan(toRadian(fov / 2.0));

    Vec3d end(x_r, y_r, z_r);


    /* Rotate */
    auto camera_angles = getAngles(direction);
    auto main_angles = getAngles({0.0, 0.0, 1.0});
    auto rotate_angles = camera_angles - main_angles;

    rotate_angles[0] = toRadian(rotate_angles[0]);
    rotate_angles[1] = toRadian(rotate_angles[1]);
    rotate_angles[2] = toRadian(rotate_angles[2]);

    auto rotate_matrix = Matrix3x3::rotateZ(rotate_angles[2]);
    end = rotate_matrix * end;
    rotate_matrix = Matrix3x3::rotateX(rotate_angles[0]);
    end = rotate_matrix * end;
    rotate_matrix = Matrix3x3::rotateY(rotate_angles[1]);
    end = rotate_matrix * end;

    /* Translate */
    end = end + position;

    return Ray(start, end);
}