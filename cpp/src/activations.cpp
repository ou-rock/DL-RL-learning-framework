#include "activations.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace lf {

Matrix Activations::sigmoid(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.size(); ++i) {
        float val = std::max(-500.0f, std::min(500.0f, x.data()[i]));
        result.data()[i] = 1.0f / (1.0f + std::exp(-val));
    }
    return result;
}

Matrix Activations::sigmoid_derivative(const Matrix& x) {
    Matrix sig = sigmoid(x);
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.size(); ++i) {
        float s = sig.data()[i];
        result.data()[i] = s * (1.0f - s);
    }
    return result;
}

Matrix Activations::relu(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = std::max(0.0f, x.data()[i]);
    }
    return result;
}

Matrix Activations::relu_derivative(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = x.data()[i] > 0.0f ? 1.0f : 0.0f;
    }
    return result;
}

Matrix Activations::softmax(const Matrix& x) {
    Matrix result(x.rows(), x.cols());

    // Row-wise softmax with numerical stability
    for (size_t r = 0; r < x.rows(); ++r) {
        // Find max in row
        float max_val = x(r, 0);
        for (size_t c = 1; c < x.cols(); ++c) {
            max_val = std::max(max_val, x(r, c));
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t c = 0; c < x.cols(); ++c) {
            float exp_val = std::exp(x(r, c) - max_val);
            result(r, c) = exp_val;
            sum += exp_val;
        }

        // Normalize
        for (size_t c = 0; c < x.cols(); ++c) {
            result(r, c) /= sum;
        }
    }
    return result;
}

Matrix Activations::tanh(const Matrix& x) {
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.size(); ++i) {
        result.data()[i] = std::tanh(x.data()[i]);
    }
    return result;
}

Matrix Activations::tanh_derivative(const Matrix& x) {
    Matrix t = tanh(x);
    Matrix result(x.rows(), x.cols());
    for (size_t i = 0; i < x.size(); ++i) {
        float tv = t.data()[i];
        result.data()[i] = 1.0f - tv * tv;
    }
    return result;
}

} // namespace lf
