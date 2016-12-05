//
// Created by Esha Uboweja on 12/5/16.
//

#include "TestBinaryTensor3D.h"


bool TestBinaryTensor3D::test_im2col_single(uint rows, uint cols, uint block_width, uint block_height, uint padding,
                                            uint stride) {
    return false;
}

bool TestBinaryTensor3D::test_im2col_invalid(uint rows, uint cols, uint block_width, uint block_height, uint padding,
                                             uint stride) {
    return false;
}

bool TestBinaryTensor3D::test_im2col() {
    return true;
}

void TestBinaryTensor3D::runAllTests() {
    std::cout << "----Testing BinaryTensor3D class functions...\n";
    bool result = test_im2col();
    std::cout << "[TestBinaryLayer] Tests completed! Result = " << (result? "PASSED" : "FAILED") << std::endl;
}