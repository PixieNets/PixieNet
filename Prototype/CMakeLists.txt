cmake_minimum_required(VERSION 3.6)
project(Prototype)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp BinaryTensor.cpp BinaryTensor.h)
add_executable(Prototype ${SOURCE_FILES})