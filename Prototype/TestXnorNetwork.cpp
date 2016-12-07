//
// Created by Zal on 12/6/16.
//

#include "TestXnorNetwork.h"
#include "BinaryConvolution.h"
#include "XnorNetwork.h"

using namespace bconv;

void TestXnorNetwork::runMiniNet() {
    // This function expects an iput with 3 channels and outputs a cube of 96 channels
    std::cout << "-- XnorNetwork MiniNet test" << std::endl;
    XnorNetwork xnorNet;
    xnorNet.buildMiniNet();

    arma::cube input = randu<arma::cube>(55,55,3);
    arma::vec output = xnorNet.forwardPass(input);
    std::cout << "[XnorNetwork::ForwardPass]: output size is: " ;
    std::cout << output.size() << std::endl;
}

bool testBinConvSize(arma::cube input, int filterSz, int filterCh, int stride, int outSz, int outCh, Convolution convType, Nonlinearity activType, Pooling poolType, uint poolSize, uint poolStride) {
    BinaryConvolution testBinConv = BinaryConvolution(filterSz, filterSz, filterCh, outCh, stride, convType, activType, poolType, poolSize, poolStride);
    arma::cube output = testBinConv.forwardPass(input);
    return false;
}

//void TestXnorNetwork::testAllBinConvSizes() {
//    arma::cube in1 = arma::randu<arma::cube>(227,227,3);
//    if( ! testBinConvSize(in1, 227, 3, 3, 55, 96, Convolution::same, Nonlinearity::relu, Pooling::max, 3,2) ) {
//        std::cout << "Failed BinConvSizes 1" << std::endl;
//    }
//
//    arma::cube in1 = arma::randu<arma::cube>(55,55,96);
//    if( ! testBinConvSize(in1, 55, 5, 3, 27, 256, Convolution::same, Nonlinearity::relu, Pooling::max, 3,2) ) {
//        std::cout << "Failed BinConvSizes 1" << std::endl;
//    }
//
//    BinaryConvolution binConv1 = BinaryConvolution(5, 5, 96, 256, 3, Convolution::same, Nonlinearity::relu, Pooling::max, 3, 2);
//    BinaryConvolution binConv2 = BinaryConvolution(3, 3, 256, 384, 1, Convolution::same, Nonlinearity::relu, Pooling::none);
//    BinaryConvolution binConv3 = BinaryConvolution(3, 3, 384, 384, 1, Convolution::same, Nonlinearity::relu, Pooling::none);
//    BinaryConvolution binConv4 = BinaryConvolution(3, 3, 384, 256, 1, Convolution::same, Nonlinearity::relu, Pooling::max, 3, 2);
//    BinaryConvolution binConv5 = BinaryConvolution(6, 6, 256, 4096, 1, Convolution::same, Nonlinearity::relu, Pooling::none);
//    BinaryConvolution binConv6 = BinaryConvolution(1, 1, 4096, 1000, 1, Convolution::same, Nonlinearity::relu, Pooling::none);
//}

void TestXnorNetwork::runAllTests() {
    this->runMiniNet();
}