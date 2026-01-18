#ifndef BACKPROP_H
#define BACKPROP_H

#include "matrix.h"
#include "activations.h"
#include <unordered_map>
#include <string>

namespace lf {

class BackpropEngine {
public:
    BackpropEngine();

    // Weight management
    void set_weights(const std::string& name, const Matrix& weights);
    Matrix get_weights(const std::string& name) const;

    // Forward pass: X -> hidden (sigmoid) -> output (softmax)
    Matrix forward(const Matrix& X);

    // Backward pass: compute gradients
    std::unordered_map<std::string, Matrix> backward(const Matrix& y_true);

    // Loss computation
    float cross_entropy_loss(const Matrix& y_pred, const Matrix& y_true) const;

private:
    std::unordered_map<std::string, Matrix> weights_;

    // Cached values from forward pass (needed for backward)
    Matrix X_cache_;
    Matrix z1_cache_;
    Matrix a1_cache_;
    Matrix z2_cache_;
    Matrix a2_cache_;
};

} // namespace lf

#endif
