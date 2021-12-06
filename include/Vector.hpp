#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <stdexcept>

template<typename T, int Dim>
class Vector {
public:
    T operator[](int i) {
        if (i >= Dim)
            throw std::runtime_error("Index is out of range");
        return m_data[i];
    }
private:
    T m_data[Dim];
};

using Vertex = Vector<double, 3>;

#endif
