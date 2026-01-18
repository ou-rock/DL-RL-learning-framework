cmake_minimum_required(VERSION 3.15)
project(learning_core_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /fp:fast /W4)
else()
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra)
endif()

# Find pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Source files
set(CORE_SOURCES
    src/matrix.cpp
    src/activations.cpp
    src/backprop.cpp
    src/optimizers.cpp
)

# Python module
pybind11_add_module(learning_core_cpp ${CORE_SOURCES} bindings/pybind11_wrapper.cpp)
target_include_directories(learning_core_cpp PRIVATE include)

# Optional: C++ tests with Google Test
option(BUILD_TESTS "Build C++ tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS learning_core_cpp DESTINATION .)
