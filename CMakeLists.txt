cmake_minimum_required(VERSION 3.20)
project(ray_tracer_cpp)

find_package (Threads)

include_directories(include)
set(CMAKE_CXX_STANDARD 17)

add_executable(ray_tracer_cpp
        src/main.cpp
        src/ray.cpp
        src/ApplicationData.cpp
        src/ApplicationLogic.cpp
        src/geometry/AGeomerty.cpp
        src/Parser.cpp
        src/Camera.cpp
        src/geometry/Plane.cpp
        src/geometry/Cylinder.cpp
        src/LightSource.cpp
        src/geometry/Sphere.cpp
        src/geometry/Square.cpp
        src/geometry/Triangle.cpp
        src/math/math_utils.cpp
        src/math/Matrix.cpp
        )

set(SFML_LIBS sfml-graphics sfml-window sfml-system)

target_link_libraries(ray_tracer_cpp ${SFML_LIBS} Threads::Threads)