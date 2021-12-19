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

Matrix3x3::Matrix3x3() {

}

Matrix3x3 Matrix3x3::rotateX(double alpha) {
    Matrix3x3 result;

    result.data[0][0] = 1;
    result.data[0][1] = 0;
    result.data[0][2] = 0;
    result.data[1][0] = 0;
    result.data[1][1] = cos(alpha);
    result.data[1][2] = -sin(alpha);
    result.data[2][0] = 0;
    result.data[2][1] = sin(alpha);
    result.data[2][2] = cos(alpha);

    return result;
}

Matrix3x3 Matrix3x3::rotateY(double beta) {
    Matrix3x3 result;

    result.data[0][0] = cos(beta);
    result.data[0][1] = 0;
    result.data[0][2] = sin(beta);
    result.data[1][0] = 0;
    result.data[1][1] = 1;
    result.data[1][2] = 0;
    result.data[2][0] = -sin(beta);
    result.data[2][1] = 0;
    result.data[2][2] = cos(beta);

    return result;
}

Matrix3x3 Matrix3x3::rotateZ(double gamma) {
    Matrix3x3 result;

    result.data[0][0] = cos(gamma);
    result.data[0][1] = -sin(gamma);
    result.data[0][2] = 0;
    result.data[1][0] = sin(gamma);
    result.data[1][1] = cos(gamma);
    result.data[1][2] = 0;
    result.data[2][0] = 0;
    result.data[2][1] = 0;
    result.data[2][2] = 1;

    return result;
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


