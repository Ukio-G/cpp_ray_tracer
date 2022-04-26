#ifndef CUDA_MATH_UTILS
#define CUDA_MATH_UTILS

#include "Vector.cuh"
#include <math.h>
#include "optional.cuh"
#include "cuPair.cuh"

__device__ __host__ double cuToRadian(double degree);
__device__ __host__ double cuToDeg(double rad);
__device__ __host__ cuVec3d cuGetAngles(cuVec3d v1);

template<class T>
__device__ __host__ T cuClamp(T v, T from, T to) {
    return (from > v) ? from : ((to < v) ? to : v);
}

template<class T, int Dim>
__device__ __host__ double cuDot(cuVector<T, Dim> a, cuVector<T, Dim> b) {
    double result = 0.0;
    for (int i = 0; i < Dim; ++i)
        result += a[i] * b[i];
    return result;
}

template <class T>
__device__ __host__ cuVector<T, 3> cuCross(cuVector<T, 3> a, cuVector<T, 3> b) {
    T i = a[1] * b[2] - a[2] * b[1];
    T j = a[2] * b[0] - a[0] * b[2];
    T k = a[0] * b[1] - a[1] * b[0];
    cuVector<T, 3> result(i, j, k);
    return result;
}

template <class T, int Dim>
__device__ __host__ cuVector<T, Dim> cuVecClamp(cuVector<T, Dim> v, T from, T to) {
    T result_data[Dim];
    
    for (int i = 0; i < Dim; ++i)
        result_data[i] = cuClamp(v[i], from, to);
    
    cuVector<T, Dim> result(result_data);
    return result;
}

template <class T, int Dim>
__device__ __host__ cuVector<T, Dim> cuReflect(cuVector<T, Dim> I, cuVector<T, Dim> N) {
    auto dot_value = cuDot(I, N) * 2;
    N = N * dot_value;
    I = I - N;
    return I;
}

template <class T, int Dim>
__device__ __host__ double cuMixedProduct(cuVector<T, Dim> v, cuVector<T, Dim> w, cuVector<T, Dim> z) {
    return cuDot(cuCross(v, w), z);
}

template <class T>
__device__ __host__ T cuMin(const T& a, const T& b) {
    return (a > b) ? b : a;
}

template <class T>
__device__ __host__ T cuMax(const T& a, const T& b) {
    return (a < b) ? b : a;
}

template<typename T>
__device__ __host__ cu::optional<cuPair<T, T>> solveSquareEq(cuVec3d cf) {
    T d = cf[1] * cf[1] - 4.0 * cf[0] * cf[2];
    if (d < 0)
        return cu::nullopt;
    d = sqrt(d);
    cuPair<T, T> result{ (-cf[1] + d) / (2.0 * cf[0]), (-cf[1] - d) / (2.0 * cf[0]) };
    return result;
}

#endif