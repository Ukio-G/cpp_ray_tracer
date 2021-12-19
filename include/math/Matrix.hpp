#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include "Vector.hpp"

class Matrix3x3 {
public:
    double data[3][3];
    Matrix3x3();

    /* Vectors are rows */
    Matrix3x3(const Vector<double, 3> & r0, const Vector<double, 3> & r1, const Vector<double, 3> & r2);

    Matrix3x3(const Matrix3x3 & other);
    Vector<double, 3> operator*(Vector<double, 3> & vector);
    Vector<double, 3> operator/(Vector<double, 3> & vector);
    Matrix3x3 operator+(Matrix3x3 & other);

    static Matrix3x3 rotateX(double alpha);
    static Matrix3x3 rotateY(double beta);
    static Matrix3x3 rotateZ(double gamma);
};


std::ostream & operator<<(std::ostream & stream, const Matrix3x3 & matrix);
#endif
