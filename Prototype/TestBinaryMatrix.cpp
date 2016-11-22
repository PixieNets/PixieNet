//
// Created by Zal on 11/20/16.
//

#include "TestBinaryMatrix.h"


using namespace std;

BinaryMatrix TestBinaryMatrix::generateDiag(int n) {
    BinaryMatrix res(n,n);
    for(int i=0; i<n; ++i) {
        res.setValueAt(i,i,1);
    }
    return res;
}

BinaryMatrix TestBinaryMatrix::generateUpperDiag(int n) {
    BinaryMatrix res(n,n);
    for(int i=0; i<n; ++i) {
        for(int j=i; j < n; ++j) {
            res.setValueAt(i,j,1);
        }
    }
    return res;
}

BinaryMatrix TestBinaryMatrix::generateLowerDiag(int n) {
    BinaryMatrix res(n,n);
    for(int i=0; i<n; ++i) {
        for(int j=i; j < n; ++j) {
            res.setValueAt(j,i,1);
        }
    }
    return res;
}

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
            printf("%u ", bMtx.getBit(bMtx.data[i],j));
        }
        cout << endl;
    }
    cout << endl;

    cout << "elem_accessor" << endl;
    cout << bMtx.toString() << endl;
    for(int i=0; i<bMtx.height; ++i) {
        for(int j=0; j<bMtx.width; ++j) {
            std::pair<int, int> pos = bMtx.elemAccessor(i*bMtx.width+j,bMtx.dataLength, bMtx.baseSize, bMtx.transposed);
            printf("[%d,%d]: ", pos.first, pos.second);
            printf("%u\n", bMtx.getBit(bMtx.data[pos.first], pos.second));
        }
    }
}

void TestBinaryMatrix::testSetBit(){
    cout << "----- TEST SET BITS" << endl;
    int testSize = 6;
    BinaryMatrix bMtx = this->generateDiag(testSize);

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

    cout << "Toggle Linear Index:" << endl;
    BinaryMatrix bMtx2(testSize, testSize,0);
    bool toggle = true;
    for(int i=0; i<testSize*testSize; ++i) {
        bMtx2.setValueAt(i,toggle? 1:0);
        toggle = !toggle;
    }
    cout << bMtx2.dataToString() << endl;
}

void TestBinaryMatrix::testTransposeIdx() {
    cout << "----- TEST TRANSPOSE INDEX" << endl;
    int testSize = 4;
    BinaryMatrix bMtx(testSize,testSize);

    for(int i=0; i<testSize*testSize; ++i) {
        printf("%d ", i);
    }
    cout << endl;
    for(int i=0; i<testSize*testSize; ++i) {
        printf("%d ", bMtx.transposeIndex(i));
    }
    cout << endl << endl;
}

void TestBinaryMatrix::testTranspose(){
    cout << "----- TEST TRANSPOSE" << endl;
    int testSize = 3;
    BinaryMatrix bMtx = this->generateUpperDiag(testSize);

    cout << "U =" << endl;
    cout << bMtx.dataToString() << endl;
    bMtx.T();
    cout << "U.T = " << endl;
    cout << bMtx.dataToString() << endl;

    cout << "getDataAccessor transposed" << endl;
    for(int i=0; i<bMtx.height; ++i) {
        for(int j=0; j<bMtx.width; ++j) {
            std::pair<int, int> pos = bMtx.getDataAccessor(i, j);
            printf("[%d,%d]: ", pos.first, pos.second);
            printf("%u\n", bMtx.getBit(bMtx.data[pos.first], pos.second));
        }
    }
}

void TestBinaryMatrix::testBinMultiply(){
    cout << "----- TEST MULTIPLY" << endl;
    int testSize = 6;
    BinaryMatrix mtx0(testSize, testSize, 0);
    BinaryMatrix mtx1(testSize, testSize, 1);

    cout << "0 = "<<endl;
    cout << mtx0.dataToString() << endl;
    cout << "1 = "<<endl;
    cout << mtx1.dataToString() << endl;
    cout << endl;

    BinaryMatrix resZeroZero = mtx0 * mtx0;
    cout << "0 x 0 = " << endl;
    cout << resZeroZero.dataToString() << endl;

    BinaryMatrix resZeroOne = mtx0 * mtx1;
    cout << "0 x 1 = " << endl;
    cout << resZeroOne.dataToString() << endl;

    BinaryMatrix resOneOne = mtx1 * mtx1;
    cout << "1 x 1 = " << endl;
    cout << resOneOne.dataToString() << endl;
}

void TestBinaryMatrix::testTBinMultiply(){
    cout << "----- TEST TRANSPOSE MULTIPLY" << endl;
    int testSize = 6;
    BinaryMatrix uDiag(testSize,testSize);
    BinaryMatrix lDiag(testSize,testSize);
    //Fill upper triangle
    for(int i=0; i<testSize; ++i) {
        for(int j=i; j < testSize; ++j) {
            uDiag.setValueAt(i,j,1);
            lDiag.setValueAt(j,i,1);
        }
    }
    cout << "uDiag = " << endl;
    cout << uDiag.dataToString() << endl;
    cout << "lDiag = " << endl;
    cout << lDiag.dataToString() << endl;

    BinaryMatrix diag = uDiag * lDiag;
    cout << "uDiag * lDiag = " << endl;
    cout << diag.dataToString() << endl;

    uDiag.T();
    BinaryMatrix lDiagT = uDiag * lDiag;
    cout << uDiag.toString() << endl;
    cout << lDiag.toString() << endl;
    cout << "uDiag.T * lDiag = " << endl;
    cout << lDiagT.dataToString() << endl;
}

void TestBinaryMatrix::testDoubleMultiply(){
    cout << "----- TEST DOUBLE MULTIPLY" << endl;
    int testSize = 3;
    BinaryMatrix uDiag(testSize,testSize);
    BinaryMatrix lDiag(testSize,testSize);
    //Fill upper triangle
    for(int i=0; i<testSize; ++i) {
        for(int j=i; j < testSize; ++j) {
            uDiag.setValueAt(i,j,1);
            lDiag.setValueAt(j,i,1);
        }
    }

    double *dMtx = new double[testSize*testSize];
    for(int i=0; i<testSize*testSize; ++i) {
        dMtx[i] = 109.0;
    }
    double *dResMtx = uDiag.doubleMultiply(dMtx);

    for(int row=0; row<testSize; ++row) {
        for(int col=0; col<testSize; ++col) {
            printf("%.1f ", dMtx[row*testSize+col]);
        }
        printf("\n");
    }
    for(int row=0; row<testSize; ++row) {
        for(int col=0; col<testSize; ++col) {
            printf("%.1f ", dResMtx[row*testSize+col]);
        }
        printf("\n");
    }

    if(dMtx!= nullptr) delete[] dMtx;
    if(dResMtx!= nullptr) delete[] dResMtx;
}

void TestBinaryMatrix::runAllTests(){
    testCreateAndPrint();
    testGetBit();
    testSetBit();
    testTransposeIdx();
    testTranspose();
    testBinMultiply();
    testTBinMultiply();
    testDoubleMultiply();
}