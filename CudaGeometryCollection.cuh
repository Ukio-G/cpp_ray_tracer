#ifndef CUDA_GEOMETRY_COLLECTION
#define CUDA_GEOMETRY_COLLECTION

#include "AGeometry.cuh"
#include "Square.cuh"
#include "Plane.cuh"
#include "Sphere.cuh"
#include "Cylinder.cuh"
#include "Triangle.cuh"

class cuGeometryCollection
{
public:
	cuAGeometry* data;

	// cuAGeometry impl objects 
	size_t geometry_count;
	
	// bytes
	size_t geometry_size;

	cuAGeometry** arrayPtrs;

	cuSphere* m_spheres_begin_ptr;
	size_t spheres_count;

	cuPlane* m_planes_begin_ptr;
	size_t planes_count;

	cuCylinder* m_cylinders_begin_ptr;
	size_t cylinders_count;

	cuTriangle* m_triangles_begin_ptr;
	size_t triangles_count;

	cuSquare* m_squares_begin_ptr;
	size_t squares_count;
};

#endif