#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "matrix.h"
#include "activations.h"
#include "backprop.h"
#include "optimizers.h"

namespace py = pybind11;
using namespace lf;

PYBIND11_MODULE(learning_core_cpp, m) {
    m.doc() = "C++ core implementations for learning framework";

    // Matrix class
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def(py::init<size_t, size_t, float>())
        .def(py::init<size_t, size_t, const std::vector<float>&>())
        .def_property_readonly("rows", &Matrix::rows)
        .def_property_readonly("cols", &Matrix::cols)
        .def_property_readonly("size", &Matrix::size)
        .def("at", py::overload_cast<size_t, size_t>(&Matrix::at))
        .def("matmul", &Matrix::matmul)
        .def("transpose", &Matrix::transpose)
        .def("add", &Matrix::add)
        .def("subtract", &Matrix::subtract)
        .def("multiply", &Matrix::multiply)
        .def("elementwise_multiply", &Matrix::elementwise_multiply)
        .def("fill", &Matrix::fill)
        .def("zeros", &Matrix::zeros)
        .def("ones", &Matrix::ones)
        .def("to_vector", &Matrix::to_vector)
        .def("to_numpy", [](const Matrix& mat) {
            return py::array_t<float>(
                {mat.rows(), mat.cols()},
                {mat.cols() * sizeof(float), sizeof(float)},
                mat.data()
            );
        })
        .def("__repr__", [](const Matrix& m) {
            return "<Matrix " + std::to_string(m.rows()) + "x" +
                   std::to_string(m.cols()) + ">";
        });

    // Activations class (static methods)
    py::class_<Activations>(m, "Activations")
        .def_static("sigmoid", &Activations::sigmoid)
        .def_static("sigmoid_derivative", &Activations::sigmoid_derivative)
        .def_static("relu", &Activations::relu)
        .def_static("relu_derivative", &Activations::relu_derivative)
        .def_static("softmax", &Activations::softmax)
        .def_static("tanh", &Activations::tanh)
        .def_static("tanh_derivative", &Activations::tanh_derivative);

    // BackpropEngine class
    py::class_<BackpropEngine>(m, "BackpropEngine")
        .def(py::init<>())
        .def("set_weights", &BackpropEngine::set_weights)
        .def("get_weights", &BackpropEngine::get_weights)
        .def("forward", &BackpropEngine::forward)
        .def("backward", &BackpropEngine::backward)
        .def("cross_entropy_loss", &BackpropEngine::cross_entropy_loss);

    // SGDMomentum optimizer
    py::class_<SGDMomentum>(m, "SGDMomentum")
        .def(py::init<float, float>(),
             py::arg("learning_rate") = 0.01f,
             py::arg("momentum") = 0.9f)
        .def_readwrite("learning_rate", &SGDMomentum::learning_rate)
        .def_readwrite("momentum", &SGDMomentum::momentum)
        .def("step", &SGDMomentum::step)
        .def("zero_velocity", &SGDMomentum::zero_velocity);

    // Adam optimizer
    py::class_<Adam>(m, "Adam")
        .def(py::init<float, float, float, float>(),
             py::arg("learning_rate") = 0.001f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("epsilon") = 1e-8f)
        .def_readwrite("learning_rate", &Adam::learning_rate)
        .def_readwrite("beta1", &Adam::beta1)
        .def_readwrite("beta2", &Adam::beta2)
        .def_readwrite("epsilon", &Adam::epsilon)
        .def("step", &Adam::step)
        .def("reset", &Adam::reset);
}
