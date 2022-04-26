#include "Camera.cuh"
#include "Vector.cuh"

__host__ __device__ cuCamera::cuCamera() { }

__host__ __device__  cuRay cuCamera::rayForPixel(unsigned int x, unsigned int y, unsigned int w, unsigned int h) {
    auto start = position;
    auto half_width = (double)w / 2.0;

    auto x_r = (double)x - half_width;
    auto y_r = (double)h / 2.0 - (double)y;
    auto z_r = half_width / tan(cuToRadian(fov / 2.0));

    cuVec3d end(-x_r, y_r, z_r);

    auto camera_angles = cuGetAngles(direction.normalized());
    auto main_angles = cuGetAngles({ 0.0, 0.0, 1.0 });
    auto rotate_angles = camera_angles - main_angles;

    rotate_angles[0] = cuToRadian(rotate_angles[0]);
    rotate_angles[1] = cuToRadian(rotate_angles[1]);
    rotate_angles[2] = cuToRadian(rotate_angles[2]);

    //auto rotate_matrix = Matrix3x3::rotateX(rotate_angles[2]);
    //end = rotate_matrix * end;
    auto rotate_matrix = cuMatrix3x3::rotateX(rotate_angles[0]);
    end = rotate_matrix * end;
    rotate_matrix = cuMatrix3x3::rotateY(rotate_angles[1]);
    end = rotate_matrix * end;

   
    end = end + position;
    return cuRay(start, end);
}

