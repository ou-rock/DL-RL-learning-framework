#include "optimizers.h"
#include <cmath>

namespace lf {

// ============== SGDMomentum ==============

SGDMomentum::SGDMomentum(float learning_rate, float momentum)
    : learning_rate(learning_rate), momentum(momentum) {}

Matrix SGDMomentum::step(const std::string& param_name, const Matrix& param, const Matrix& grad) {
    // Initialize velocity if first time
    auto it = velocity_.find(param_name);
    if (it == velocity_.end()) {
        velocity_.emplace(param_name, Matrix(param.rows(), param.cols(), 0.0f));
        it = velocity_.find(param_name);
    }

    Matrix& vel = it->second;

    // velocity = momentum * velocity - lr * grad
    for (size_t i = 0; i < vel.size(); ++i) {
        vel.data()[i] = momentum * vel.data()[i] - learning_rate * grad.data()[i];
    }

    // param = param + velocity
    return param.add(vel);
}

void SGDMomentum::zero_velocity() {
    velocity_.clear();
}


// ============== Adam ==============

Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon)
    : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

Matrix Adam::step(const std::string& param_name, const Matrix& param, const Matrix& grad) {
    // Initialize if first time
    if (m_.find(param_name) == m_.end()) {
        m_.emplace(param_name, Matrix(param.rows(), param.cols(), 0.0f));
        v_.emplace(param_name, Matrix(param.rows(), param.cols(), 0.0f));
        t_[param_name] = 0;
    }

    Matrix& m = m_.at(param_name);
    Matrix& v = v_.at(param_name);
    int& t = t_.at(param_name);

    t += 1;

    // Update biased first moment estimate
    // m = beta1 * m + (1 - beta1) * grad
    for (size_t i = 0; i < m.size(); ++i) {
        m.data()[i] = beta1 * m.data()[i] + (1.0f - beta1) * grad.data()[i];
    }

    // Update biased second moment estimate
    // v = beta2 * v + (1 - beta2) * grad^2
    for (size_t i = 0; i < v.size(); ++i) {
        float g = grad.data()[i];
        v.data()[i] = beta2 * v.data()[i] + (1.0f - beta2) * g * g;
    }

    // Bias correction
    float beta1_t = std::pow(beta1, static_cast<float>(t));
    float beta2_t = std::pow(beta2, static_cast<float>(t));

    // Compute update
    Matrix result(param.rows(), param.cols());
    for (size_t i = 0; i < result.size(); ++i) {
        float m_hat = m.data()[i] / (1.0f - beta1_t);
        float v_hat = v.data()[i] / (1.0f - beta2_t);
        result.data()[i] = param.data()[i] - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    return result;
}

void Adam::reset() {
    m_.clear();
    v_.clear();
    t_.clear();
}

} // namespace lf
