cmake_minimum_required(VERSION 3.6)
project(Prototype)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp BinaryMatrix.cpp BinaryMatrix.h BinaryLayer.cpp BinaryLayer.h TestBinaryMatrix.cpp TestBinaryMatrix.h)
add_executable(Prototype ${SOURCE_FILES})