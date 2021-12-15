#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include "Vector.hpp"

template<class T, int Dim>
double dot(Vector<T, Dim> a, Vector<T, Dim> b) {
    double result = 0.0;
    for (int i = 0; i < Dim; ++i)
        result += a[i] * b[i];
    return result;
}

template <class T>
Vector<T, 3> cross(Vector<T, 3> a, Vector<T, 3> b) {
    T i = a[1] * b[2] - a[2]*b[1];
    T j = a[2] * b[0] - a[0]*b[2];
    T k = a[0] * b[1] - a[1]*b[0];
    Vector<T, 3> result(i, j, k);
    return result;
}

double toRadian(double degree);

template <class T, int Dim>
Vector<T, Dim> clamp(Vector<T, Dim> v, T from, T to) {
    T result_data[Dim];
    for (int i = 0; i < Dim; ++i)
        result_data[i] = std::clamp(v[i], from, to);
    Vector<T, Dim> result(result_data);
    return result;
}

template <class T, int Dim>
Vector<T, Dim> reflect(Vector<T, Dim> I, Vector<T, Dim> N) {
    return I - 2.0 * dot(N, I) * N;
}

template <class T, int Dim>
double mixedProduct(Vector<T, Dim> v, Vector<T, Dim> w, Vector<T, Dim> z) {
    return dot(cross(v, w), z);
}

#endif
