#include "vec3.h"
#ifndef _COLOR_H
#define _COLOR_H

class Color
{
private:
	double r;
	double g;
	double b;
public:
	Color();
	Color(double R, double G, double B);
	~Color();
	Color operator+(const Color& c) const;
	Color operator-(const Color& c) const;
	Color operator*(double a) const;
	Color operator/(double a) const;
	const Color& operator=(const Color& c);
	double& operator[](int i);
	friend std::ostream& operator<<(std::ostream& out, Color& c);

	double R() const;
	double G() const;
	double B() const;
};

#endif // !_COLOR_H

