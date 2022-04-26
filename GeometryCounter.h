#pragma once
#include <string>
#include <fstream>

class GeometryCounter
{
public:
	GeometryCounter();

	void countFromFile(const std::string& filename);

	unsigned int getSpheresCount() const;
	unsigned int getPlanesCount() const;
	unsigned int getTrianglesCount() const;
	unsigned int getCylindersCount() const;
	unsigned int getSquaresCount() const;

	unsigned int getShapesCount() const;

private:
	void increaseCounters();

	char m_line_buf[512]; // line buffer

	unsigned int m_spheres;
	unsigned int m_planes;
	unsigned int m_triangles;
	unsigned int m_cylinders;
	unsigned int m_squares;
};

