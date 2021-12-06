#include "color.h"

Color::Color() : r(0), g(0), b(0) {}

Color::Color(double R, double G, double B) : r(R), g(G), b(B) {}

Color::~Color() {};

Color Color::operator+(const Color& c)const {
	return Color(r + c.R(), g + c.G(), b + c.B());
}

Color Color::operator-(const Color& c)const {
	return Color(r - c.R(), g - c.G(), b - c.B());
}

Color Color::operator*(double a) const {
	return Color(r * a, g * a, b * a);
}

Color Color::operator/(double a) const {
	return Color(r / a, g / a, b / a);
}

const Color& Color::operator=(const Color& c) {
	if (this != &c) {
		r = c.R();
		g = c.G();
		b = c.B();
	}
	return *this;
}

double& Color::operator[](int i) {
	if (i == 0)
		return r;
	if (i == 1)
		return g;
    return b;
}

std::ostream& operator<<(std::ostream& out, Color& c) {
	out << c[0] << ' ' << c[1] << ' ' << c[2];
	return out;
}

double Color::R() const {
	return r;
}

double Color::G() const {
	return g;
}

double Color::B() const {
	return b;
}