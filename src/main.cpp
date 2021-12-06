#include <iostream>
#include "vec3.h"
#include "color.h"
#include "ray.h"

vec3 unit_vector(vec3 v) {
    return v /v.length();
}

double DotProduct(vec3 v1, vec3 v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

bool hit_sphere(const Ray& ray, const vec3& C, double radius) {
    vec3 D = ray.Direction();
    vec3 OC = ray.Origin() - C;
    double a = DotProduct(D, D);
    double b = 2.0 * DotProduct(OC, D);
    double c = DotProduct(OC, OC);
    double discriminant = b * b - 4 * a * c;
    return (discriminant > 0);
}

Color ray_color(const Ray& r) {
    if (hit_sphere(r, vec3(0, 0, 2.0), 1.0))
        return Color(1.0, 0, 0);
    vec3 unit_direction = unit_vector(r.Direction());
    double t = 0.5 * (unit_direction.Y() + 1.0);
    return Color(1.0, 1.0, 1.0) * (1.0 - t) + Color(0.5, 0.7, 1.0) * t;
}


int main()
{
    int image_width = 800;
    int image_height = 600;

    double Vh = 3;
    double Vw = 4;
    double d = 1.0;

    vec3 O = vec3(0, 0, 0);
    vec3 horizontal = vec3(Vw, 0, 0);
    vec3 vertical = vec3(0, Vh, 0);
    vec3 lower_left_corner = O - horizontal / 2 - vertical / 2 + vec3(0, 0, d);

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            double u = double(i) / (image_width - 1);
            double v = double(j) / (image_height - 1);
            vec3 V = lower_left_corner + horizontal * u + vertical * v;
            Ray r = Ray(O, V - O);
            Color pixel_color = ray_color(r);
            int ir = static_cast<int>(255.999 * pixel_color[0]);
            int ig = static_cast<int>(255.999 * pixel_color[1]);
            int ib = static_cast<int>(255.999 * pixel_color[2]);
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    std::cerr << "\nDone.\n";


    //
    /*int Vw = 4;
    int Vh = 3;
    int Cw = 800;
    int Ch = 600;
    double d = 1.0;
    vec3 O = vec3(0, 0, 0);

    std::cout << "P3\n" << Cw << ' ' << Ch << "\n255\n";

    for (int Cx = -400; Cx < 400; Cx++) {
        for (int Cy = -300; Cy < 300; Cy++) {
            vec3 V = vec3(Cx * double(Vw / Cw), Cy * double(Vh / Ch), d);
            vec3 D = V - O;
            Ray r = Ray(O, D);
            Color pixel_color = ray_color(r);
            int ir = static_cast<int>(255.999 * pixel_color.R());
            int ig = static_cast<int>(255.999 * pixel_color.G());
            int ib = static_cast<int>(255.999 * pixel_color.B());
            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }*/

	return 0;
}