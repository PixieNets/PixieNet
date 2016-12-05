//
// Created by Esha Uboweja on 12/5/16.
//

#include "BinaryTensor3D.h"
#include "TestBinaryTensor3D.h"

using namespace bd;

#define DEBUG 1

void TestBinaryTensor3D::printTestArma2(std::string testName, std::string desc, arma::mat input) {
#ifdef DEBUG
    std::cout << "[TestBinaryTensor3D::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryTensor3D::printTestUArma2(std::string testName, std::string desc, arma::umat input) {
#ifdef DEBUG
    std::cout << "[TestBinaryTensor3D::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryTensor3D::printTestBM(std::string testName, std::string desc, BinaryMatrix input) {
#ifdef DEBUG
    std::cout << "[TestBinaryTensor3D::" << testName << "] binary tensor3d " << desc << " : \n";
    input.print();
    std::cout << std::endl;
#endif
}

void TestBinaryTensor3D::printTestArma3(std::string testName, std::string desc, arma::cube input) {
#ifdef DEBUG
    std::cout << "[TestBinaryTensor3D::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryTensor3D::printTestUArma3(std::string testName, std::string desc, arma::ucube input) {
#ifdef DEBUG
    std::cout << "[TestBinaryTensor3D::" << testName << "] arma cube " << desc << " : \n" << input << std::endl;
#endif
}

void TestBinaryTensor3D::printTestBT3(std::string testName, std::string desc, BinaryTensor3D input) {
#ifdef DEBUG
    std::cout << "[TestBinaryTensor3D::" << testName << "] binary tensor3d " << desc << " : \n"
              << input.toString() << std::endl;
#endif
}

bool TestBinaryTensor3D::test_im2col_single(uint rows, uint cols, uint channels, uint block_width, uint block_height,
                                            uint padding, uint stride) {

    std::string testName = "test_im2col_single";
    // Generate a random binary matrix
    arma::ucube input3D = BinaryTensor3D::randomArmaUCube(rows, cols, channels);
    BinaryTensor3D bt3(input3D);

    printTestUArma3(testName, "input", input3D);
    printTestBT3(testName, "input", bt3);

    // Compare im2col result for binary matrix and arma
    arma::umat armaResult = BinaryTensor3D::im2colArma(input3D, block_width, block_height, padding, stride);
    printTestUArma2(testName, "output", armaResult);

    BinaryLayer blResult = bt3.im2col(block_width, block_height, padding, stride);
    printTestBM(testName, "output", *(blResult.binMtx()));

    return blResult.binMtx()->equalsArmaMat(armaResult);
}

bool TestBinaryTensor3D::test_im2col_invalid(uint rows, uint cols, uint channels, uint block_width, uint block_height,
                                             uint padding, uint stride) {
    std::string testName = "test_im2col_invalid";
    // Generate a random binary matrix
    arma::ucube input3D = BinaryTensor3D::randomArmaUCube(rows, cols, channels);
    BinaryTensor3D bt3(input3D);

    printTestUArma3(testName, "input", input3D);
    printTestBT3(testName, "input", bt3);

    try {
        BinaryLayer blResult = bt3.im2col(block_width, block_height, padding, stride);
        printTestBM(testName, "output", *(blResult.binMtx()));
    } catch (std::exception e) {
        return true;
    }

    std::cerr << "[TestBinaryTensor3D::test_im2col_invalid] Test didn't raise exception\n";
    // didn't raise an exception
    return false;
}

bool TestBinaryTensor3D::test_im2col() {
    return test_im2col_single()
        && test_im2col_single(3, 3, 2)
        && test_im2col_single(3, 3, 7)
        && test_im2col_single(3, 3, 10, 3, 3, 1, 1)
        && test_im2col_single(3, 3, 8, 3, 3, 1, 2)
        && test_im2col_single(5, 5, 6, 3, 3, 1, 2)
        && test_im2col_single(7, 9, 2, 5, 5, 2, 1)
        && test_im2col_invalid(8, 6, 3, 3, 3, 1, 2)
        && test_im2col_invalid(7, 9, 2, 5, 5, 3, 1)
        && test_im2col_invalid(10, 10, 1, 3, 3, 0, 2);
}

void TestBinaryTensor3D::runAllTests() {
    std::cout << "----Testing BinaryTensor3D class functions...\n";
    bool result = test_im2col();
    std::cout << "[TestBinaryLayer] Tests completed! Result = " << (result? "PASSED" : "FAILED") << std::endl;
}