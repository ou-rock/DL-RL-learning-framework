#include "backprop.h"
#include <cmath>
#include <stdexcept>

namespace lf {

BackpropEngine::BackpropEngine()
    : X_cache_(1, 1), z1_cache_(1, 1), a1_cache_(1, 1),
      z2_cache_(1, 1), a2_cache_(1, 1) {}

void BackpropEngine::set_weights(const std::string& name, const Matrix& weights) {
    weights_[name] = weights;
}

Matrix BackpropEngine::get_weights(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

Matrix BackpropEngine::forward(const Matrix& X) {
    // Cache input
    X_cache_ = X;

    // Layer 1: X @ W1 -> sigmoid
    const Matrix& W1 = weights_.at("W1");
    z1_cache_ = X.matmul(W1);
    a1_cache_ = Activations::sigmoid(z1_cache_);

    // Layer 2: a1 @ W2 -> softmax
    const Matrix& W2 = weights_.at("W2");
    z2_cache_ = a1_cache_.matmul(W2);
    a2_cache_ = Activations::softmax(z2_cache_);

    return a2_cache_;
}

std::unordered_map<std::string, Matrix> BackpropEngine::backward(const Matrix& y_true) {
    std::unordered_map<std::string, Matrix> grads;

    size_t batch_size = X_cache_.rows();
    float batch_scale = 1.0f / static_cast<float>(batch_size);

    // dL/dz2 = (a2 - y) / batch_size
    Matrix dz2 = a2_cache_.subtract(y_true).multiply(batch_scale);

    // dL/dW2 = a1.T @ dz2
    Matrix a1_T = a1_cache_.transpose();
    Matrix dW2 = a1_T.matmul(dz2);
    grads["W2"] = dW2;

    // dL/da1 = dz2 @ W2.T
    const Matrix& W2 = weights_.at("W2");
    Matrix W2_T = W2.transpose();
    Matrix da1 = dz2.matmul(W2_T);

    // dL/dz1 = da1 * sigmoid'(z1)
    Matrix sig_deriv = Activations::sigmoid_derivative(z1_cache_);
    Matrix dz1 = da1.elementwise_multiply(sig_deriv);

    // dL/dW1 = X.T @ dz1
    Matrix X_T = X_cache_.transpose();
    Matrix dW1 = X_T.matmul(dz1);
    grads["W1"] = dW1;

    return grads;
}

float BackpropEngine::cross_entropy_loss(const Matrix& y_pred, const Matrix& y_true) const {
    float loss = 0.0f;
    size_t batch_size = y_true.rows();

    for (size_t i = 0; i < y_true.rows(); ++i) {
        for (size_t j = 0; j < y_true.cols(); ++j) {
            if (y_true(i, j) > 0.5f) {  // One-hot encoded
                loss -= std::log(y_pred(i, j) + 1e-10f);
            }
        }
    }

    return loss / static_cast<float>(batch_size);
}

} // namespace lf
