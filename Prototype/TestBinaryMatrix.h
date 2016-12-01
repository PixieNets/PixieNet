//
// Created by Zal on 11/20/16.
//

#ifndef PROTOTYPE_TESTBINARYMATRIX_H
#define PROTOTYPE_TESTBINARYMATRIX_H

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

    bool test_im2col();

public:
    bool runAllTests();
};


#endif //PROTOTYPE_TESTBINARYMATRIX_H
