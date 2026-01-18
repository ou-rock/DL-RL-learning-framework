#include "matrix.h"
#include <cstring>

namespace lf {

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(new float[rows * cols]) {}

Matrix::Matrix(size_t rows, size_t cols, float value)
    : rows_(rows), cols_(cols), data_(new float[rows * cols]) {
    fill(value);
}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<float>& data)
    : rows_(rows), cols_(cols), data_(new float[rows * cols]) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size mismatch");
    }
    std::copy(data.begin(), data.end(), data_);
}

Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_), data_(new float[other.size()]) {
    std::copy(other.data_, other.data_ + size(), data_);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = new float[size()];
        std::copy(other.data_, other.data_ + size(), data_);
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.data_ = nullptr;
    other.rows_ = other.cols_ = 0;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        other.data_ = nullptr;
        other.rows_ = other.cols_ = 0;
    }
    return *this;
}

Matrix::~Matrix() { delete[] data_; }

float& Matrix::at(size_t r, size_t c) {
    if (r >= rows_ || c >= cols_) throw std::out_of_range("Index out of range");
    return data_[r * cols_ + c];
}

const float& Matrix::at(size_t r, size_t c) const {
    if (r >= rows_ || c >= cols_) throw std::out_of_range("Index out of range");
    return data_[r * cols_ + c];
}

Matrix Matrix::matmul(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Incompatible dimensions for matmul");
    }
    Matrix result(rows_, other.cols_, 0.0f);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t k = 0; k < cols_; ++k) {
            float a_ik = at(i, k);
            for (size_t j = 0; j < other.cols_; ++j) {
                result(i, j) += a_ik * other(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = at(i, j);
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Dimension mismatch for add");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Dimension mismatch for subtract");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::multiply(float scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix Matrix::elementwise_multiply(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Dimension mismatch");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

void Matrix::fill(float value) {
    std::fill(data_, data_ + size(), value);
}

std::vector<float> Matrix::to_vector() const {
    return std::vector<float>(data_, data_ + size());
}

} // namespace lf
