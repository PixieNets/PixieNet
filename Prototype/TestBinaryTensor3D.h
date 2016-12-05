//
// Created by Esha Uboweja on 12/5/16.
//

#pragma once

#include "BinaryTensor3D.h"

using namespace bd;

class TestBinaryTensor3D {
private:

    void printTestUArma2(std::string testName, std::string desc, arma::umat input);
    void printTestArma2(std::string testName, std::string desc, arma::mat input);
    void printTestBM(std::string testName, std::string desc, BinaryMatrix input);

    void printTestUArma3(std::string testName, std::string desc, arma::ucube input);
    void printTestArma3(std::string testName, std::string desc, arma::cube input);
    void printTestBT3(std::string testName, std::string desc, BinaryTensor3D input);

    bool test_im2col_single(uint rows = 10, uint cols = 10, uint channels = 1,
                            uint block_width = 3, uint block_height = 3,
                            uint padding = 0, uint stride = 1);
    bool test_im2col_invalid(uint rows, uint cols, uint channels,
                             uint block_width, uint block_height,
                             uint padding, uint stride);
    bool test_im2col();

public:
    void runAllTests();
};

