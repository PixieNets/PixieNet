//
// Created by Zal on 12/6/16.
//

#include "TestXnorNetwork.h"
#include "BinaryConvolution.h"

using namespace bconv;

void TestXnorNetwork::runMiniNet() {
    // This function expects an iput with 3 channels and outputs a cube of 96 channels
    // If the input is 227, it is expectec to output a wxh=55x55
    BinaryConvolution testBinConv = BinaryConvolution(10, 10, 3, 96, 1, Convolution::same, Nonlinearity::relu, Pooling::max, 2, 2);
    arma::cube testInput = arma::randu<arma::cube>(100,100,3);
    arma::cube testRes = testBinConv.forwardPass(testInput);
    std::cout << "Output size:" << std::endl;
    std::cout <<  testRes.size() << std::endl;
    printf("[TestXnorNetwork::runMiniNet] testRes size = (%llu, %llu, %llu) \n", testRes.n_rows, testRes.n_cols, testRes.n_slices);
}

void TestXnorNetwork::runAllTests() {
    this->runMiniNet();
}