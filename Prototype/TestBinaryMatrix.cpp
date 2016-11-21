//
// Created by Zal on 11/20/16.
//

#include "TestBinaryMatrix.h"
#include "BinaryMatrix.h"

using namespace std;
void TestBinaryMatrix::testCreateAndPrint(){
    cout << "----- TEST CONSTRUCTOR" << endl;
    int testSize = 5;
    cout << "Default ctor:" << endl;
    BinaryMatrix bMtx(testSize, testSize);
    cout << "Data Length: " << bMtx.dataLength << endl;
    cout << bMtx.toString() << endl;
    cout << bMtx.dataToString() << endl;

    cout << "Init 0 ctor:" << endl;
    BinaryMatrix bMtx0(testSize, testSize,0);
    cout << "Data Length: " << bMtx0.dataLength << endl;
    cout << bMtx0.toString() << endl;
    cout << bMtx0.dataToString() << endl;

    cout << "Init 1 ctor:" << endl;
    BinaryMatrix bMtx1(testSize, testSize, 1);
    cout << "Data Length: " << bMtx1.dataLength << endl;
    cout << bMtx1.toString() << endl;
    cout << bMtx1.dataToString() << endl;
}

void TestBinaryMatrix::testGetBit(){
    cout << "----- TEST GET BITS" << endl;
    int testSize = 4;
    cout << "get_bit" << endl;
    BinaryMatrix bMtx(testSize, testSize, 1);
    cout << bMtx.toString() << endl;
    for(int i=0; i<bMtx.dataLength; ++i) {
        printf("%u ",bMtx.data[i]);
        for(int j=0; j<bMtx.baseSize; ++j) {
            printf("%u ", bMtx.get_bit(bMtx.data[i],j));
        }
        cout << endl;
    }
    cout << endl;

    cout << "elem_accessor" << endl;
    cout << bMtx.toString() << endl;
    for(int i=0; i<bMtx.height; ++i) {
        for(int j=0; j<bMtx.width; ++j) {
            std::pair<int, int> pos = bMtx.elem_accessor(i*bMtx.width+j,bMtx.dataLength, bMtx.baseSize, bMtx.transposed);
            printf("[%d,%d]: ", pos.first, pos.second);
            printf("%u\n", bMtx.get_bit(bMtx.data[pos.first], pos.second));
        }
    }
}

void TestBinaryMatrix::testSetBit(){
    cout << "----- TEST SET BITS" << endl;
    int testSize = 6;
    BinaryMatrix bMtx(testSize, testSize, 0);
    for(int i=0; i<testSize; ++i) {
        bMtx.setValueAt(i,i,1);
    }
    cout << bMtx.toString() << endl;
    cout << bMtx.dataToString() << endl;

    BinaryMatrix bMtx1(testSize, testSize,0);
    for(int i=0; i<testSize; ++i) {
        bMtx1.setValueAt(0,i,1);
        bMtx1.setValueAt(testSize-1,i,1);
        bMtx1.setValueAt(i,0,1);
        bMtx1.setValueAt(i,testSize-1,1);
    }
    cout << bMtx1.toString() << endl;
    cout << bMtx1.dataToString() << endl;
}

void TestBinaryMatrix::testTranspose(){
    cout << "----- TEST TRANSPOSE" << endl;
    int testSize = 3;
    BinaryMatrix bMtx(testSize,testSize);

    //Fill upper triangle
    for(int i=0; i<testSize; ++i) {
        for(int j=i; j < testSize; ++j) {
            bMtx.setValueAt(i,j,1);
        }
    }
    cout << bMtx.dataToString() << endl;
    bMtx.T();
    cout << bMtx.dataToString() << endl;

    cout << "getDataAccessor transposed" << endl;
    for(int i=0; i<bMtx.height; ++i) {
        for(int j=0; j<bMtx.width; ++j) {
            std::pair<int, int> pos = bMtx.getDataAccessor(i, j);
            printf("[%d,%d]: ", pos.first, pos.second);
            printf("%u\n", bMtx.get_bit(bMtx.data[pos.first], pos.second));
        }
    }
}

void TestBinaryMatrix::testBinMultiply(){
    cout << "----- TEST Multiply" << endl;
    int testSize = 5;
    BinaryMatrix mtx0(testSize, testSize, 0);
    BinaryMatrix mtx1(testSize, testSize, 1);

    cout << mtx0.dataToString() << endl;
    cout << mtx1.dataToString() << endl;
    cout << endl;

    BinaryMatrix resZero = mtx0 * mtx1;
    cout << "0 x 0 = " << endl;
    cout << resZero.toString() << endl;
    cout << resZero.dataToString() << endl;

    BinaryMatrix resOne = mtx1 * mtx1;
    cout << "1 x 1 = " << endl;
    cout << resOne.toString() << endl;
    cout << resOne.dataToString() << endl;

    

}

void TestBinaryMatrix::testTBinMultiply(){

}

void TestBinaryMatrix::testDoubleMultiply(){

}

void TestBinaryMatrix::runAllTests(){
    testCreateAndPrint();
    testGetBit();
    testSetBit();
    testTranspose();
    testBinMultiply();
    testTBinMultiply();
    testDoubleMultiply();
}