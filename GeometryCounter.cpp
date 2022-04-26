#include "GeometryCounter.h"

GeometryCounter::GeometryCounter() : 
    m_spheres(0), 
    m_planes(0), 
    m_triangles(0), 
    m_cylinders(0), 
    m_squares(0) {
    memset(m_line_buf, 0, 512);
}

void GeometryCounter::countFromFile(const std::string& filename) {
    std::fstream file(filename);
    memset(m_line_buf, 0, 512);

    while (file.getline(m_line_buf, 512)) {
        increaseCounters();
        memset(m_line_buf, 0, 512);
    }
}

unsigned int GeometryCounter::getSpheresCount() const {
    return m_spheres;
}

unsigned int GeometryCounter::getPlanesCount() const {
    return m_planes;
}

unsigned int GeometryCounter::getTrianglesCount() const {
    return m_triangles;
}

unsigned int GeometryCounter::getCylindersCount() const {
    return m_cylinders;
}

unsigned int GeometryCounter::getSquaresCount() const {
    return m_squares;
}

unsigned int GeometryCounter::getShapesCount() const
{
    return m_spheres + m_planes + m_cylinders + m_squares + m_triangles;
}

void GeometryCounter::increaseCounters() {
    if (strncmp(m_line_buf, "sp", 2) == 0)
        m_spheres++;
    if (strncmp(m_line_buf, "pl", 2) == 0)
        m_planes++;
    if (strncmp(m_line_buf, "cy", 2) == 0)
        m_cylinders++;
    if (strncmp(m_line_buf, "sq", 2) == 0)
        m_squares++;
    if (strncmp(m_line_buf, "tr", 2) == 0)
        m_triangles++;
}


