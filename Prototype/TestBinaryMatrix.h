//
// Created by Zal on 11/20/16.
//

#ifndef PROTOTYPE_TESTBINARYMATRIX_H
#define PROTOTYPE_TESTBINARYMATRIX_H

#include <iostream>

class TestBinaryMatrix {
    void testCreateAndPrint();
    void testGetBit();
    void testSetBit();
    void testSetValueAt();
    void testTransposeIdx();
    void testTranspose();
    void testBinMultiply();
    void testTBinMultiply();
    void testDoubleMultiply();

public:
    void runAllTests();
};


#endif //PROTOTYPE_TESTBINARYMATRIX_H
