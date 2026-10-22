cmake_minimum_required(VERSION 3.18)
project(cricket_vision_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cricket_vision_cpp trajectory.cpp)

target_compile_definitions(cricket_vision_cpp PRIVATE PYTHON_MODULE)

set_target_properties(cricket_vision_cpp PROPERTIES
    PREFIX ""
    OUTPUT_NAME "cricket_vision_cpp"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../cricket_vision/cpp"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../cricket_vision/cpp"
)
