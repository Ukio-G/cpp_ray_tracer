#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#include "AGeomerty.hpp"

class Triangle : public AGeomerty {
public:
    Triangle();
    Triangle(Color color_, Vertex a, Vertex b, Vertex c);
    Vertex vertexes[3];
};


#endif
