cmake_minimum_required(VERSION 3.20)
project(ray_tracer_cpp)


include_directories(include)
set(CMAKE_CXX_STANDARD 17)

add_executable(ray_tracer_cpp
        src/main.cpp
        src/Color.cpp
        src/ray.cpp
        src/vec3.cpp
        src/ApplicationData.cpp
        src/ApplicationLogic.cpp
        src/AGeomerty.cpp
        src/Parser.cpp
        src/Camera.cpp
        src/Plane.cpp
        src/Cylinder.cpp
        src/LightSource.cpp
        src/Sphere.cpp
        src/Square.cpp
        src/Triangle.cpp
        )
