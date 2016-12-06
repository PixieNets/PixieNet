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

void TestBinaryConvolution::printTestUArma4(std::string testName, std::string desc, ArmaUTensor4D input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] arma tensor4D" << desc << " : \n";
    for (uint f = 0; f < input.size(); ++f) {
        std::cout << "[ARMA FILTER #" << f << "]:\n" << input[f] << std::endl;
    }
#endif
}

void TestBinaryConvolution::printTestBT4(std::string testName, std::string desc, BinaryTensor4D input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] binary tensor4d " << desc << " : \n"
              << BinaryConvolution::bt4ToString(input) << std::endl;
#endif
}

template<typename T>
void TestBinaryConvolution::printTestVec(std::string testName, std::string desc, std::vector<T> input) {
#ifdef DEBUG
    std::cout << "[TestBinaryConvolution::" << testName << "] vector " << desc << " : \n";
    for (auto value : input) {
        std::cout << value << std::endl;
    }
    std::cout << std::endl;
#endif
}

bool TestBinaryConvolution::test_convolution_single(uint rows_in, uint cols_in, uint width, uint height,
                                                    uint channels, uint filters, uint stride,
                                                    Convolution conv_type) {

    std::string testName = "test_convolution_single";

    BinaryConvolution bconv = BinaryConvolution(width, height, channels, filters, stride, conv_type, Nonlinearity::none,
                                                Pooling::none);

    // Generate a random input matrix
    arma::cube input3D = BinaryTensor3D::randomArmaCube(rows_in, cols_in, channels);
    printTestArma3(testName, "Arma Input3D", input3D);

    // Normalize the input
    arma::cube norm_input3D = bconv.normalizeData3D(input3D);
    printTestArma3(testName, "Arma Normalized Input3D", norm_input3D);

    // Compute K beta matrix
    arma::mat K = bconv.input2KMat(norm_input3D);
    printTestArma2(testName, "Arma K betas for input", K);

    // Generate a random weights matrix
    ArmaUTensor4D armaWeights4D = BinaryConvolution::randomTensor4DUArma(width, height, channels, filters);
    printTestUArma4(testName, "Arma weights 4D", armaWeights4D);

    std::vector<double> alphaPerFilter;
    alphaPerFilter.reserve(filters);
    for (uint f = 0; f < filters; ++f) {
        alphaPerFilter.emplace_back(1.0);
    }
    printTestVec(testName, "weights 4D alphas", alphaPerFilter);

    BinaryTensor4D bt4Weights4D = BinaryConvolution::uarmaToBT4(armaWeights4D);
    printTestBT4(testName, "Binary weights 4D", bt4Weights4D);
    bconv.setWeights(bt4Weights4D);




    return true;
}

bool TestBinaryConvolution::test_convolution() {
    return test_convolution_single();
}

void TestBinaryConvolution::runAllTests() {
    std::cout << "----Testing BinaryConvolution class functions...\n";
    bool result = test_convolution();
    std::cout << "[TestBinaryConvolution] Tests completed! Result = " << (result? "PASSED" : "FAILED") << std::endl;
}