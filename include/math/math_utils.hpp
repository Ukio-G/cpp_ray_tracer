#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include "Vector.hpp"

template<class T, int Dim>
double dot(Vector<T, Dim> & a, Vector<T, Dim> & b) {
    double result = 0.0;
    for (int i = 0; i < Dim; ++i)
        result += a[i] * b[i];
    return result;
}

template <class T>
Vector<T, 3> cross(Vector<T, 3> & a, Vector<T, 3> & b) {
    T i = a[1] * b[2] - a[2]*b[1];
    T j = a[2] * b[0] - a[0]*b[2];
    T k = a[0] * b[1] - a[1]*b[0];
    Vector<T, 3> result(i, j, k);
    return result;
}

double toRadian(double degree);

#endif
