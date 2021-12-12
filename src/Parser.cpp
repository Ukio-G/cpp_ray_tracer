#include "Parser.hpp"
#include <sstream>

std::optional<Parser::ParseItem> Parser::parse_line(const std::string &line) {
    if (auto result = parseGeometry(line))      return result;
    if (auto result = parseCamera(line))        return result;
    if (auto result = parseLight(line))         return result;
    if (auto result = parseWindowSize(line))    return result;

    return std::nullopt;
}

std::shared_ptr<Camera> Parser::parseCamera(const std::string &line) {
    if (line[0] != 'C')
        return nullptr;

    std::string type, position, direction, fov;
    std::istringstream iss(line);

    iss >> type >> position >> direction >> fov;

    return std::make_shared<Camera>(Camera(parseVector(position), parseVector(direction), std::atof(fov.c_str())));
}

/*   Light   */
std::shared_ptr<ALight> Parser::parseLight(const std::string &line) {
    if (line[0] == 'A')
        return std::make_shared<AmbientLight>(parseAmbientLight(line));
    if (line[0] == 'L')
        return std::make_shared<LightSource>(parseLightSource(line));
    return nullptr;
}

AmbientLight Parser::parseAmbientLight(const std::string &line) {
    std::string type, ratio, color;
    std::istringstream iss(line);

    iss >> type >> ratio >> color;
    return AmbientLight(parseColor(color), std::atof(ratio.c_str()));
}

LightSource Parser::parseLightSource(const std::string &line) {
    std::string type, position, ratio, color;
    std::istringstream iss(line);

    iss >> type >> position >> ratio >> color;
    return LightSource(parseColor(color), std::atof(ratio.c_str()), parseVector(position));
}

std::shared_ptr<Vec2i> Parser::parseWindowSize(const std::string &line) {
    if (line[0] != 'W')
        return nullptr;
    std::string type;
    int w, h;
    std::istringstream iss(line);

    iss >> type >> w >> h;
    return std::make_shared<Vec2i>(Vec2i(w, h));
}

/*   Geometry   */

std::shared_ptr<AGeomerty> Parser::parseGeometry(const std::string &line) {
    if (strncmp(line.c_str(), "sp", 2) == 0)
        return std::make_shared<Sphere>(parseSphere(line));
    if (strncmp(line.c_str(), "pl", 2) == 0)
        return std::make_shared<Plane>(parsePlane(line));
    if (strncmp(line.c_str(), "cy", 2) == 0)
        return std::make_shared<Cylinder>(parseCylinder(line));
//    if (strncmp(line.c_str(), "sq", 2) == 0)
//        return std::make_shared<Square>(parseSquare(line));
    if (strncmp(line.c_str(), "tr", 2) == 0)
        return std::make_shared<Triangle>(parseTriangle(line));
    return nullptr;
}

Sphere Parser::parseSphere(const std::string &line) {
    std::string type, position, radius, color;
    std::istringstream iss(line);

    iss >> type >> position >> radius >> color;

    return {parseColor(color), parseVector(position), std::atof(radius.c_str())};
}

Triangle Parser::parseTriangle(const std::string &line) {
    std::string type, v1, v2, v3, color;
    std::istringstream iss(line);

    iss >> type >> v1 >> v2 >> v2 >> color;

    return {parseColor(color), parseVector(v1), parseVector(v2), parseVector(v3)};
}

Cylinder Parser::parseCylinder(const std::string &line) {
    std::string type, position, direction, diameter, height, color;
    std::istringstream iss(line);

    iss >> type >> position >> direction >> diameter >> height >> color;

    return {parseColor(color), std::atof(diameter.c_str()), std::atof(height.c_str()), parseVector(position), parseVector(direction)};
}

Square Parser::parseSquare(const std::string &line) { // sq 0.0,0.0,-10.0 0.0,1.0,0.0 10 0,0,225
    std::string type, position, normal, size_size, color;
    std::istringstream iss(line);

    iss >> type >> position >> normal >> size_size >> color;

    return Square(parseColor(color), parseVector(position), parseVector(normal), std::atof(size_size.c_str()));
}

Plane Parser::parsePlane(const std::string &line) {
    std::string type, position, normal, color;
    std::istringstream iss(line);

    iss >> type >> position >> normal >> color;

    return Plane(parseColor(color), parseVector(position), parseVector(normal));
}

Vec3d Parser::parseVector(const std::string &line) {
    std::istringstream ss(line);
    std::string token;
    Vec3d result;
    int idx = 0;

    while(std::getline(ss, token, ','))
        result[idx++] = std::atof(token.c_str());
    return result;
}

Color Parser::parseColor(const std::string &line) {
    std::istringstream ss(line);
    std::string token;
    Color result;
    int idx = 0;

    while(std::getline(ss, token, ','))
        result[idx++] = std::atof(token.c_str());
    return result;
}