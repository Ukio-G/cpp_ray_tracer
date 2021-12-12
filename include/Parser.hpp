#ifndef PARSER_HPP
#define PARSER_HPP

#include <memory>
#include <variant>
#include <optional>
#include <geometry/Square.hpp>
#include <geometry/Plane.hpp>
#include <geometry/Cylinder.hpp>
#include <geometry/Triangle.hpp>
#include <geometry/Sphere.hpp>
#include "geometry/AGeomerty.hpp"
#include "Camera.hpp"
#include "LightSource.hpp"
#include "Vector.hpp"
#include "ALight.hpp"
#include "AmbientLight.hpp"
#include <vector>
#include "SceneComponents.hpp"

class Parser {
public:
    using ParseItem = std::variant<std::shared_ptr<AGeomerty>, std::optional<Camera>, std::shared_ptr<ALight>, std::optional<Vec2i>>;

    std::optional<ParseItem> parseLine(const std::string & line);
    SceneComponents parseFile(const std::string & filename);

private:
    std::shared_ptr<AGeomerty>  parseGeometry(const std::string & line);
    std::optional<Camera>       parseCamera(const std::string & line);
    std::shared_ptr<ALight>     parseLight(const std::string & line);
    std::optional<Vec2i>        parseWindowSize(const std::string & line);

    /* Geometry parser */
    Sphere      parseSphere(const std::string & line);
    Triangle    parseTriangle(const std::string & line);
    Cylinder    parseCylinder(const std::string & line);
    Square      parseSquare(const std::string & line);
    Plane       parsePlane(const std::string & line);

    /* Light parser */
    LightSource     parseLightSource(const std::string & line);
    AmbientLight    parseAmbientLight(const std::string & line);

    Vec3d parseVector(const std::string & line);
    Color parseColor(const std::string & line);
};


#endif
