//
// Created by Esha Uboweja on 12/5/16.
//

#pragma once

#include "BinaryConvolution.h"

using namespace bconv;

class TestBinaryConvolution {
private:
    void printTestUArma2(std::string testName, std::string desc, arma::umat input);
    void printTestArma2(std::string testName, std::string desc, arma::mat input);
    void printTestBM(std::string testName, std::string desc, BinaryMatrix input);

    void printTestUArma3(std::string testName, std::string desc, arma::ucube input);
    void printTestArma3(std::string testName, std::string desc, arma::cube input);
    void printTestBT3(std::string testName, std::string desc, BinaryTensor3D input);

    void printTestUArma4(std::string testName, std::string desc, ArmaUTensor4D input);
    void printTestBT4(std::string testName, std::string desc, BinaryTensor4D input);

    template<typename T>
    void printTestVec(std::string testName, std::string desc, std::vector<T> input);


    bool test_convolution_single(uint rows_in = 3, uint cols_in = 3, uint width = 3,
                                 uint height = 3, uint channels = 2, uint filters = 1,
                                 uint stride = 1, Convolution conv_type=Convolution::same);
    bool test_convolution();

public:
    void runAllTests();
};

