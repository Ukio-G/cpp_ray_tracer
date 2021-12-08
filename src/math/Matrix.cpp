#include <math/Matrix.hpp>

Vector<double, 3> Matrix3x3::operator*(Vector<double, 3> &vector) {
    double a = vector[0] * data[0][0] + vector[1] * data[0][1] + vector[2] * data[0][2];
    double b = vector[0] * data[1][0] + vector[1] * data[1][1] + vector[2] * data[1][2];
    double c = vector[0] * data[2][0] + vector[1] * data[2][1] + vector[2] * data[2][2];

    Vector<double, 3> result(a, b, c);
    return result;
}

Vector<double, 3> Matrix3x3::operator/(Vector<double, 3> &vector) {
    Vector<double, 3> inv(1.0/vector[0], 1.0/vector[1], 1.0/vector[2]);
    return (*this * inv);
}

Matrix3x3 Matrix3x3::operator+(Matrix3x3 &other) {
    Matrix3x3 result = *this;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result.data[i][j] += other.data[i][j];
    return result;
}

Matrix3x3::Matrix3x3(const Vector<double, 3> &r0, const Vector<double, 3> &r1, const Vector<double, 3> &r2) {
    memcpy(&data[0][0], r0.data(), sizeof(double ) * 3);
    memcpy(&data[1][0], r1.data(), sizeof(double ) * 3);
    memcpy(&data[2][0], r2.data(), sizeof(double ) * 3);
}

Matrix3x3::Matrix3x3(const Matrix3x3 &other) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            data[i][j] = other.data[i][j];
}

std::ostream & operator<<(std::ostream & stream, const Matrix3x3 & matrix) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            stream << matrix.data[i][j] << " ";
        }
        stream << "\n";
    }
    return stream;
}
