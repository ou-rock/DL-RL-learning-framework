#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "matrix.h"
#include <unordered_map>
#include <string>

namespace lf {

class SGDMomentum {
public:
    SGDMomentum(float learning_rate = 0.01f, float momentum = 0.9f);

    float learning_rate;
    float momentum;

    Matrix step(const std::string& param_name, const Matrix& param, const Matrix& grad);
    void zero_velocity();

private:
    std::unordered_map<std::string, Matrix> velocity_;
};


class Adam {
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f,
         float beta2 = 0.999f, float epsilon = 1e-8f);

    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;

    Matrix step(const std::string& param_name, const Matrix& param, const Matrix& grad);
    void reset();

private:
    std::unordered_map<std::string, Matrix> m_;  // First moment
    std::unordered_map<std::string, Matrix> v_;  // Second moment
    std::unordered_map<std::string, int> t_;     // Timestep per parameter
};

} // namespace lf

#endif
