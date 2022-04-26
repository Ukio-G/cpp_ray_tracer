#ifndef CUDA_VECTOR
#define CUDA_VECTOR

#include <cuda_runtime.h>

#define CUDOOP(op) __device__ __host__ cuVector<T,Dim> operator op(const cuVector &other) const { cuVector<T,Dim> result; for (int i = 0; i < Dim; ++i) result.m_data[i] = m_data[i] op other.m_data[i]; return result; }

/* Forward declaration */
template<typename T, int Dim> class cuVector;
template<class T, int Dim> __device__ double cuDot(cuVector<T, Dim> a, cuVector<T, Dim> b);

template<typename T, int Dim>
class cuVector {
public:
    __host__ __device__ cuVector() {
        memset(m_data, 0, Dim * sizeof(T));
    }

    template <class ...A>
    __host__ __device__ cuVector(A... args) {
        T input[] = { args... };
        for (int i = 0; i < Dim; ++i)
            m_data[i] = input[i];
    }

    __host__ __device__ cuVector(T* array) {
        for (int i = 0; i < Dim; ++i)
            m_data[i] = array[i];
    }

    __host__ __device__ cuVector<T, Dim> normalized() {
        double locLength = length();
        double inv_length = (1 / locLength);

        T nomalized_dimensions[Dim];

        for (int i = 0; i < Dim; ++i)
            nomalized_dimensions[i] = m_data[i] * inv_length;
        return cuVector(nomalized_dimensions);
    }

    __host__ __device__ cuVector<T, Dim>& normalize() {
        *this = this->normalized();
    }

    __host__ __device__ cuVector<T, Dim> addLengthToNewVec(double len) {
        cuVector<T, Dim> result = *this;

        result.normalized();
        result = result * len;
        result = result + *this;

        return result;
    }

    __host__ __device__ cuVector<T, Dim>& addLength(double len) {
        *this = addLengthToNewVec(len);
    }

    // Работает только с векторами, которые не помечены как const
    __host__ __device__ T& operator[](int i) {
        return m_data[i];
    }
    
    __host__ __device__ T operator[](int i) const {
        return m_data[i];
    }

    CUDOOP(+);
    CUDOOP(-);

    __host__ __device__ cuVector<T, Dim> operator*(T value) {
        cuVector<T, Dim> result;
        for (int i = 0; i < Dim; ++i)
            result.m_data[i] = m_data[i] * value;
        return result;
    }

    __host__ __device__ cuVector<T, Dim> operator/(T value) {
        cuVector<T, Dim> result;
        for (int i = 0; i < Dim; ++i)
            result.m_data[i] = m_data[i] / value;
        return result;
    }

    __host__ __device__ unsigned int getDimension() {
        return Dim;
    }

    __host__ __device__ double length() {
        double result = cuDot(*this, *this);
        return std::sqrt(result);
    }

    __host__ __device__ void setLength(double length) {
        normalize();
        *this = *this * length;
    }

    __host__ __device__ cuVector<T, Dim> inverse() {
        cuVector<T, Dim> result = *this;

        for (int i = 0; i < Dim; ++i)
            result.m_data[i] = -result.m_data[i];

        return result;
    }

    __host__ __device__ const T* data() const { return reinterpret_cast<const double*>(&m_data); }

    __host__ __device__ static cuVector<T, Dim> vectorFromPoints(cuVector<T, Dim> a, cuVector<T, Dim> b) {
        return b - a;
    }
private:
    T m_data[Dim];
};

using cuVertex = cuVector<double, 3>;
using cuColor = cuVector<double, 3>;
using cuVec3d = cuVector<double, 3>;
using cuVec2i = cuVector<int, 2>;


#endif