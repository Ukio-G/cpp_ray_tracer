#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "AGeomerty.hpp"
#include <ostream>

class Triangle : public AGeomerty {
public:
    Triangle();
    Triangle(Color color_, Vertex a, Vertex b, Vertex c);
    Vertex vertexes[3];
    bool intersect(const Ray & ray);
};

inline std::ostream & operator<<(std::ostream &ostream, Triangle & triangle) {
    ostream << "color: " << triangle.color;
    ostream << "vertexes: ";

    for (int i = 0; i < 3; ++i) {
        ostream << "(" << triangle.vertexes[i][0]
                << "," << triangle.vertexes[i][1]
                << "," << triangle.vertexes[i][2] << ") ";
    }

    return ostream;
}
#endif
