#pragma once

#include "CudaGeometryCollection.cuh"
#include "ApplicationData.hpp"
#include <iostream>
#include "cuIntersection.cuh"

inline bool checkStructSize() {
	bool isSpheresSizeOk = sizeof(Sphere) == sizeof(cuSphere);
	bool isPlanesSizeOk = sizeof(Plane) == sizeof(cuPlane);
	bool isSquaresSizeOk = sizeof(Square) == sizeof(cuSquare);
	bool isTrianglesSizeOk = sizeof(Triangle) == sizeof(cuTriangle);
	bool isCylindersSizeOk = sizeof(Cylinder) == sizeof(cuCylinder);
	bool isLightSizeOk = sizeof(LightSource) == sizeof(cuLight);
	bool isAmbientLightSizeOk = sizeof(AmbientLight) == sizeof(cuAmbientLight);


	if (!isSpheresSizeOk)
		std::cout << "Error spheres size! sizeof(Sphere): " << sizeof(Sphere) << ", sizeof(cuSphere): " << sizeof(cuSphere) << std::endl;
	else
		std::cout << "Sphere size OK!" << std::endl;

	if (!isPlanesSizeOk)
		std::cout << "Error spheres size! sizeof(Plane): " << sizeof(Plane) << ", sizeof(cuPlane): " << sizeof(cuPlane) << std::endl;
	else
		std::cout << "Plane size OK!" << std::endl;

	if (!isSquaresSizeOk)
		std::cout << "Error spheres size! sizeof(Square): " << sizeof(Square) << ", sizeof(cuSquare): " << sizeof(cuSquare) << std::endl;
	else
		std::cout << "Square size OK!" << std::endl;

	if (!isTrianglesSizeOk)
		std::cout << "Error spheres size! sizeof(Triangle): " << sizeof(Triangle) << ", sizeof(cuTriangle): " << sizeof(cuTriangle) << std::endl;
	else
		std::cout << "Triangle size OK!" << std::endl;

	if (!isCylindersSizeOk)
		std::cout << "Error spheres size! sizeof(Cylinder): " << sizeof(Cylinder) << ", sizeof(cuCylinder): " << sizeof(cuCylinder) << std::endl;
	else
		std::cout << "Cylinder size OK!" << std::endl;

	if (!isLightSizeOk)
		std::cout << "Error Light size! sizeof(LightSource): " << sizeof(LightSource) << ", sizeof(cuLight): " << sizeof(cuLight) << std::endl;
	else
		std::cout << "Light size OK!" << std::endl;

	if (!isAmbientLightSizeOk)
		std::cout << "Error Ambient Light size! sizeof(AmbientLight): " << sizeof(AmbientLight) << ", sizeof(cuAmbientLight): " << sizeof(cuAmbientLight) << std::endl;
	else
		std::cout << "Ambient Light size OK!" << std::endl;
	

	std::cout << "Cuda intersection struct: " << sizeof(cuIntersection) << std::endl;

	return true;
}