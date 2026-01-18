#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"

namespace lf {

class Activations {
public:
    // Forward passes
    static Matrix sigmoid(const Matrix& x);
    static Matrix relu(const Matrix& x);
    static Matrix softmax(const Matrix& x);  // Row-wise softmax
    static Matrix tanh(const Matrix& x);

    // Backward passes (derivatives)
    static Matrix sigmoid_derivative(const Matrix& x);
    static Matrix relu_derivative(const Matrix& x);
    static Matrix tanh_derivative(const Matrix& x);
};

} // namespace lf

#endif
