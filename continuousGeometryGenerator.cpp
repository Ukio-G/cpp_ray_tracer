#include "continuousGeometryGenerator.h"

ContinuousGeometryGenerator::ContinuousGeometryGenerator() :
    data(0), m_size(0),
    m_spheres_begin_ptr(0), m_spheres_insert_ptr(0), 
    m_planes_begin_ptr(0), m_planes_insert_ptr(0),
    m_cylinders_begin_ptr(0), m_cylinders_insert_ptr(0),
    m_triangles_begin_ptr(0), m_triangles_insert_ptr(0),
    m_squares_begin_ptr(0), m_squares_insert_ptr(0) { }

void ContinuousGeometryGenerator::allocateMemory(const GeometryCounter & geometryCounter) {
    size_t spheres_count = geometryCounter.getSpheresCount();
    size_t squares_count = geometryCounter.getSquaresCount();
    size_t cylinders_count = geometryCounter.getCylindersCount();
    size_t planes_count = geometryCounter.getPlanesCount();
    size_t triangles_count = geometryCounter.getTrianglesCount();

    size_t spheres_mem = spheres_count * sizeof(Sphere);
    size_t squares_mem = squares_count * sizeof(Square);
    size_t cylinders_mem = cylinders_count * sizeof(Cylinder);
    size_t planes_mem = planes_count * sizeof(Plane);
    size_t triangles_mem = triangles_count * sizeof(Triangle);
    
    m_size = spheres_mem + squares_mem + cylinders_mem + planes_mem + triangles_mem;

    if (m_size > 0) {
        data = malloc(m_size);

        if (!data)
            throw std::runtime_error("Error while allocate memory");

        m_spheres_insert_ptr = m_spheres_begin_ptr = (Sphere*)data;
        m_squares_insert_ptr = m_squares_begin_ptr = (Square*)(m_spheres_begin_ptr + spheres_count);
        m_cylinders_insert_ptr = m_cylinders_begin_ptr = (Cylinder*)(m_squares_begin_ptr + squares_count);
        m_planes_insert_ptr = m_planes_begin_ptr = (Plane*)(m_cylinders_begin_ptr + cylinders_count);
        m_triangles_insert_ptr = m_triangles_begin_ptr = (Triangle*)(m_planes_begin_ptr + planes_count);
    }

    if (spheres_count == 0)     m_spheres_insert_ptr   = m_spheres_begin_ptr   = nullptr;
    if (squares_count == 0)     m_squares_insert_ptr   = m_squares_begin_ptr   = nullptr;
    if (cylinders_count == 0)   m_cylinders_insert_ptr = m_cylinders_begin_ptr = nullptr;
    if (planes_count == 0)      m_planes_insert_ptr    = m_planes_begin_ptr    = nullptr;
    if (triangles_count == 0)   m_triangles_insert_ptr = m_triangles_begin_ptr = nullptr;
}

void ContinuousGeometryGenerator::appendSphere(const Sphere& sp) {
    memcpy(m_spheres_insert_ptr++, &sp, sizeof(Sphere));
}

void ContinuousGeometryGenerator::appendPlane(const Plane& pl) {
    memcpy(m_planes_insert_ptr++, &pl, sizeof(Plane));
}

void ContinuousGeometryGenerator::appendSquare(const Square& sq) {
    memcpy(m_squares_insert_ptr++, &sq, sizeof(Square));
}

void ContinuousGeometryGenerator::appendTriangle(const Triangle& tr) {
    memcpy(m_triangles_insert_ptr++, &tr, sizeof(Triangle));
}

void ContinuousGeometryGenerator::appendCylinder(const Cylinder& cy) {
    memcpy(m_cylinders_insert_ptr++, &cy, sizeof(Cylinder));
}

Sphere* ContinuousGeometryGenerator::getSpheresPtr()
{
    return m_spheres_begin_ptr;
}

Plane* ContinuousGeometryGenerator::getPlanesPtr()
{
    return m_planes_begin_ptr;
}

Cylinder* ContinuousGeometryGenerator::getCylindersPtr()
{
    return m_cylinders_begin_ptr;
}

Triangle* ContinuousGeometryGenerator::getTrianglesPtr()
{
    return m_triangles_begin_ptr;
}

Square* ContinuousGeometryGenerator::getSquaresPtr()
{
    return m_squares_begin_ptr;
}

size_t ContinuousGeometryGenerator::getSize()
{
    return m_size;
}

void* ContinuousGeometryGenerator::getData()
{
    return data;
}

