#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace lf {

class Matrix {
public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, float value);
    Matrix(size_t rows, size_t cols, const std::vector<float>& data);

    // Rule of five
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    ~Matrix();

    // Accessors
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    float* data() { return data_; }
    const float* data() const { return data_; }

    // Element access
    float& at(size_t r, size_t c);
    const float& at(size_t r, size_t c) const;
    float& operator()(size_t r, size_t c) { return data_[r * cols_ + c]; }
    const float& operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }

    // Operations
    Matrix matmul(const Matrix& other) const;
    Matrix transpose() const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(float scalar) const;
    Matrix elementwise_multiply(const Matrix& other) const;

    // Utilities
    void fill(float value);
    void zeros() { fill(0.0f); }
    void ones() { fill(1.0f); }
    std::vector<float> to_vector() const;

private:
    size_t rows_;
    size_t cols_;
    float* data_;
};

} // namespace lf

#endif
