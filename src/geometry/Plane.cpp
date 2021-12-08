#include "geometry/Plane.hpp"

Plane::Plane(Color color_, Vector<double, 3> position_, Vector<double, 3> direction_) : AGeomerty(color_), position(position_), direction(direction_) { }

Plane::Plane() : AGeomerty(), position({0.0, 0.0, 0.0}), direction({0.0, 0.0, 0.0}) { }
