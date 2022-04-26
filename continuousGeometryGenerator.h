#pragma once
#include "GeometryCounter.h"
#include <geometry/Sphere.hpp>
#include <geometry/Plane.hpp>
#include <geometry/Cylinder.hpp>
#include <geometry/Triangle.hpp>
#include <geometry/Square.hpp>


/*
 * Geometry order in memory
 * 
 * |___________|___________|___________|___________|___________|
 *   Spheres     Squares     Cylinder	 Planes	     Triangles
 */
class ContinuousGeometryGenerator
{
public:
	ContinuousGeometryGenerator();
	void allocateMemory(const GeometryCounter& geometryCounter);
	void appendSphere(const Sphere& sp);
	void appendPlane(const Plane& sp);
	void appendCylinder(const Cylinder& sp);
	void appendTriangle(const Triangle& sp);
	void appendSquare(const Square& sp);

	Sphere*		getSpheresPtr();
	Plane*		getPlanesPtr();
	Cylinder*	getCylindersPtr();
	Triangle*	getTrianglesPtr();
	Square*		getSquaresPtr();

	// Get size in bytes of continuous memory block
	size_t		getSize();
	void*		getData();

private:
	void* data;

	// Size in bytes of continuous memory block
	size_t m_size;

	Sphere*		m_spheres_begin_ptr;
	Sphere*		m_spheres_insert_ptr;

	Plane*		m_planes_begin_ptr;
	Plane*		m_planes_insert_ptr;

	Cylinder*	m_cylinders_begin_ptr;
	Cylinder*	m_cylinders_insert_ptr;

	Triangle*	m_triangles_begin_ptr;
	Triangle*	m_triangles_insert_ptr;

	Square*		m_squares_begin_ptr;
	Square*		m_squares_insert_ptr;
};

