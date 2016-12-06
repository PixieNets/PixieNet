//
// Created by Esha Uboweja on 12/5/16.
//

#include "BinaryLayer.h"
#include "TestBinaryConvolution.h"

#define DEBUG 1

using namespace bd;
using namespace bconv;

void TestBinaryConvolution::printTestArma2(std::string testName, std::string desc, arma::mat input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryConvolution::printTestUArma2(std::string testName, std::string desc, arma::umat input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryConvolution::printTestBM(std::string testName, std::string desc, BinaryMatrix input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] binary tensor3d " << desc << " : \n";
    input.print();
    std::cout << std::endl;
#endif
}

void TestBinaryConvolution::printTestArma3(std::string testName, std::string desc, arma::cube input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryConvolution::printTestUArma3(std::string testName, std::string desc, arma::ucube input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryConvolution::printTestBT3(std::string testName, std::string desc, BinaryTensor3D input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] binary tensor3d " << desc << " : \n"
              << input.toString() << std::endl;
#endif
}

void TestBinaryConvolution::printTestBT4(std::string testName, std::string desc, BinaryTensor4D input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] binary tensor4d " << desc << " : \n"
              << BinaryConvolution::bt4ToString(input) << std::endl;
#endif
}

void TestBinaryConvolution::runAllTests() {
    std::cout << "----Testing BinaryConvolution class functions...\n";
    bool result = false;
    std::cout << "[TestBinaryConvolution] Tests completed! Result = " << (result? "PASSED" : "FAILED") << std::endl;
}