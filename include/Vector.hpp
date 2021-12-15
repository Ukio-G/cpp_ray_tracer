#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <stdexcept>
#include <cmath>
#include <cstring>
#include <string>
#include <ostream>
#include <algorithm>

#define DOOP(op) Vector<T,Dim> operator op(const Vector &other)  { Vector<T,Dim> result; for (int i = 0; i < Dim; ++i) result.m_data[i] = m_data[i] op other.m_data[i]; return result; }

/* Forward declaration */
template<typename T, int Dim> class Vector;
template<class T, int Dim> double dot(Vector<T, Dim> a, Vector<T, Dim> b);

template<typename T, int Dim>
class Vector {
public:
    Vector() {
        memset(m_data, 0, Dim * sizeof(T));
    }

    template <class ...A>
    Vector(A... args) {
        T input[] = {args...};
        for (int i = 0; i < Dim; ++i)
            m_data[i] = input[i];
    }

    Vector(T * array) {
        for (int i = 0; i < Dim; ++i)
            m_data[i] = array[i];
    }

    Vector<T, Dim> normalized() {
        double locLength = length();
        double inv_length = (1 / locLength);

        T nomalized_dimensions[Dim];

        for (int i = 0; i < Dim; ++i)
            nomalized_dimensions[i] = m_data[i] * inv_length;
        return Vector(nomalized_dimensions);
    }

    Vector<T, Dim>& normalize() {
        *this = this->normalized();
    }

    Vector<T, Dim> addLengthToNewVec(double len) {
        Vector<T, Dim> result = *this;

        result.normalized();
        result = result * len;
        result = result + *this;

        return result;
    }

    Vector<T, Dim>& addLength(double len) {
        *this = addLengthToNewVec(len);
    }

        // Работает только с векторами, которые не помечены как const
    T& operator[](int i) {
        if (i >= Dim)
            throw std::runtime_error("Index is out of range");
        return m_data[i];
    }

    /*
    T operator[](int i) const {
        if (i >= Dim)
            throw std::runtime_error("Index is out of range");
        return m_data[i];
    }
     */

    DOOP(+);
    DOOP(-);

    Vector<T, Dim> operator*(T value) {
        Vector<T,Dim> result;
        for (int i = 0; i < Dim; ++i)
            result.m_data[i] = m_data[i] * value;
        return result;
    }

    Vector<T, Dim> operator/(T value) {
        Vector<T,Dim> result;
        for (int i = 0; i < Dim; ++i)
            result.m_data[i] = m_data[i] / value;
        return result;
    }

    unsigned int getDimension() {
        return Dim;
    }

    double length() {
        double result = dot(*this, *this);
        return std::sqrt(result);
    }

    void setLength(double length) {
        normalize();
        *this = *this * length;
    }

    Vector<T, Dim> inverse() {
        Vector<T, Dim> result = *this;

        for (int i = 0; i < Dim; ++i)
            result.m_data[i] = -result.m_data[i];

        return result;
    }

    const T * data() const {return reinterpret_cast<const double *>(&m_data);}

    static Vector<T, Dim> vectorFromPoints(Vector<T, Dim> a, Vector<T, Dim> b) {
        return b - a;
    }
private:
    T m_data[Dim];
};

using Vertex = Vector<double, 3>;
using Color = Vector<double, 3>;
using Vec3d = Vector<double, 3>;
using Vec2i = Vector<int, 2>;

inline std::ostream& operator<< (std::ostream& ostream, Vec3d& vec3D) {
    ostream << "(" << vec3D[0] << ", " << vec3D[1] << ", " << vec3D[2] << ")";
    return ostream;
}

inline std::ostream& operator<< (std::ostream& ostream, Vec2i & vec2I) {
    ostream << "(" << (int)vec2I[0] << ", " << (int)vec2I[1] << ")";
    return ostream;
}
#endif
