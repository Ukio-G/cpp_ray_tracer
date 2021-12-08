#include "geometry/AGeomerty.hpp"

AGeomerty::AGeomerty() : color({static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<uint8_t>(0)}) { }

AGeomerty::AGeomerty(Color &color_) : color(color_) { }
