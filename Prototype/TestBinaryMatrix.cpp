//
// Created by Zal on 11/20/16.
//

#include "TestBinaryMatrix.h"
#include "BinaryMatrix.h"

using namespace std;
void TestBinaryMatrix::testCreateAndPrint(){
    cout << "Test: create and build" << endl;
    cout << "Default ctor:" << endl;
    BinaryMatrix bMtx(5,5);
    cout << "Data Length: " << bMtx.dataLength << endl;
    cout << bMtx.toString() << endl;
    cout << bMtx.dataToString() << endl;
    cout << endl;
    cout << "Init 0 ctor:" << endl;
    BinaryMatrix bMtx0(5,5,0);
    cout << "Data Length: " << bMtx0.dataLength << endl;
    cout << bMtx0.toString() << endl;
    cout << bMtx0.dataToString() << endl;
    cout << endl;
    cout << "Init 1 ctor:" << endl;
    BinaryMatrix bMtx1(5,5, 1);
    cout << "Data Length: " << bMtx1.dataLength << endl;
    cout << bMtx1.toString() << endl;
    cout << bMtx1.dataToString() << endl;
    cout << endl;
}

void TestBinaryMatrix::testSetBit(){
    cout << "Test: set bits" << endl;
    int testSize = 5;
    BinaryMatrix bMtx(testSize, testSize);
    for(int i=0; i<testSize; ++i) {
        bMtx.setValueAt(i,i,1);
    }
    cout << bMtx.toString() << endl;
    cout << bMtx.dataToString() << endl;
    cout << endl;
    BinaryMatrix bMtx1(testSize, testSize,0);
    for(int i=0; i<testSize; ++i) {
        bMtx1.setValueAt(0,i,1);
        bMtx1.setValueAt(testSize-1,i,1);
        bMtx1.setValueAt(i,0,1);
        bMtx1.setValueAt(i,testSize-1,1);
    }
    cout << bMtx1.toString() << endl;
    cout << bMtx1.dataToString() << endl;
    cout << endl;
}

void TestBinaryMatrix::testGetBit(){

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