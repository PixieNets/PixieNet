//
// Created by Zal on 11/20/16.
//

#pragma once

#include <iostream>
#include "BinaryMatrix.h"

class TestBinaryMatrix {
    // Generate
    BinaryMatrix generateDiag(uint n);
    BinaryMatrix generateUpperDiag(uint n);
    BinaryMatrix generateLowerDiag(uint n);

    // Tests
    void testCreateAndPrint();
    void testGetBit();
    void testSetBit();
    void testSetValueAt();
    void testTransposeIdx();
    void testTranspose();
    void testBinMultiply();
    void testTBinMultiply();
    void testDoubleMultiply();

    bool test_initWithArma_single(uint rows = 10, uint cols = 10);
    bool test_im2col_single(uint rows = 10, uint cols = 10,
                            uint block_width = 3, uint block_height = 3,
                            uint padding = 0, uint stride = 1);
    bool test_im2col_invalid(uint rows, uint cols,
                             uint block_width, uint block_height,
                             uint padding, uint stride);

    bool test_initWithArma();
    bool test_im2col();

public:
    bool runAllTests();
};

