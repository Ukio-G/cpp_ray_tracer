#ifndef PARSER_HPP
#define PARSER_HPP

#include <memory>
#include <variant>
#include "geometry/AGeomerty.hpp"
#include "Camera.hpp"
#include "LightSource.hpp"
#include "Vector.hpp"

class Parser {
public:
    using ParsetItem = std::variant<std::shared_ptr<AGeomerty>, std::shared_ptr<Camera>, std::shared_ptr<LightSource>, Vector<int, 2>>;

    ParsetItem parse_line(const std::string & line);
};


#endif
