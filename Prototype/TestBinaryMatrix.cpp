//
// Created by Zal on 11/20/16.
//

#include "TestBinaryMatrix.h"
#include "BinaryMatrix.h"

using namespace std;
void TestBinaryMatrix::testCreateAndPrint(){
    cout << "Test: create and build" << endl;
    BinaryMatrix bMtx(2,2);
    cout << bMtx.toString() << endl;
    cout << bMtx.dataToString() << endl;
}

void TestBinaryMatrix::testGetBit(){

}

void TestBinaryMatrix::testSetBit(){

}

void TestBinaryMatrix::testGetValueAt(){

}

void TestBinaryMatrix::testBinMultiply(){

}

void TestBinaryMatrix::testTBinMultiply(){

}

void TestBinaryMatrix::testDoubleMultiply(){

}

void TestBinaryMatrix::runAllTests(){
    testCreateAndPrint();
    testGetBit();
    testSetBit();
    testGetValueAt();
    testBinMultiply();
    testTBinMultiply();
    testDoubleMultiply();
}