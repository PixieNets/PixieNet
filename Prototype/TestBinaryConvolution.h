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

    void printTestBT4(std::string testName, std::string desc, BinaryTensor4D input);

public:
    void runAllTests();
};

