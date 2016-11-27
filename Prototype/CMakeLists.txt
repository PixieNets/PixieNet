cmake_minimum_required(VERSION 3.6)
project(Prototype)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp BinaryMatrix.cpp BinaryMatrix.h BinaryLayer.cpp BinaryLayer.h TestBinaryMatrix.cpp TestBinaryMatrix.h XnorNet.cpp XnorNet.h XnorNetUtils.cpp XnorNetUtils.h TestArmadillo.cpp TestArmadillo.h BinaryConvolution.cpp BinaryConvolution.h)

add_executable(Prototype ${SOURCE_FILES})
target_link_libraries(Prototype ${ARMADILLO_LIBRARIES})
