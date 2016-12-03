//
// Created by Zal on 11/20/16.
//

#include "TestBinaryMatrix.h"
#include <armadillo>


using namespace std;
using namespace arma;

BinaryMatrix TestBinaryMatrix::generateDiag(uint n) {
    BinaryMatrix res(n,n);
    for(uint i=0; i<n; ++i) {
        res.setValueAt(i,i,1);
    }
    return res;
}

BinaryMatrix TestBinaryMatrix::generateUpperDiag(uint n) {
    BinaryMatrix res(n,n);
    for(uint i=0; i<n; ++i) {
        for(uint j=i; j < n; ++j) {
            res.setValueAt(i,j,1);
        }
    }
    return res;
}

BinaryMatrix TestBinaryMatrix::generateLowerDiag(uint n) {
    BinaryMatrix res(n,n);
    for(uint i=0; i<n; ++i) {
        for(uint j=i; j < n; ++j) {
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
    cout << "Data Length: " << bMtx.dataLength() << endl;
    cout << bMtx.toString() << endl;
    cout << bMtx.dataToString() << endl;

    cout << "Init 0 ctor:" << endl;
    BinaryMatrix bMtx0(testSize, testSize,BIT_ZERO);
    cout << "Data Length: " << bMtx0.dataLength() << endl;
    cout << bMtx0.toString() << endl;
    cout << bMtx0.dataToString() << endl;

    cout << "Init 1 ctor:" << endl;
    BinaryMatrix bMtx1(testSize, testSize, BIT_ONE);
    cout << "Data Length: " << bMtx1.dataLength() << endl;
    cout << bMtx1.toString() << endl;
    cout << bMtx1.dataToString() << endl;
}

void TestBinaryMatrix::testGetBit(){
    cout << "----- TEST GET BITS" << endl;
    int testSize = 4;
    cout << "get_bit" << endl;
    BinaryMatrix bMtx(testSize, testSize, BIT_ONE);
    cout << bMtx.toString() << endl;
    for(uint i=0; i<bMtx.dataLength(); ++i) {
        printf("%u ",bMtx.data()[i]);
        for(uint j=0; j<bMtx.baseBitSize(); ++j) {
            printf("%u ", bMtx.getBit(bMtx.data()[i],j));
        }
        cout << endl;
    }
    cout << endl;

    cout << "elem_accessor" << endl;
    cout << bMtx.toString() << endl;
    for(uint i=0; i<bMtx.height(); ++i) {
        for(uint j=0; j<bMtx.width(); ++j) {
            std::pair<uint, uint> pos = bMtx.elemAccessor(i*bMtx.width()+j,bMtx.dataLength(), bMtx.baseBitSize(), bMtx.transposed());
            printf("[%d,%d]: ", pos.first, pos.second);
            printf("%u\n", bMtx.getBit(bMtx.data()[pos.first], pos.second));
        }
    }
}

void TestBinaryMatrix::testSetBit(){
    cout << "----- TEST SET BITS" << endl;
    uint testSize = 6;
    BinaryMatrix bMtx = this->generateDiag(testSize);

    cout << bMtx.toString() << endl;
    cout << bMtx.dataToString() << endl;

    BinaryMatrix bMtx1(testSize, testSize, BIT_ZERO);
    for(uint i=0; i<testSize; ++i) {
        bMtx1.setValueAt(0,i,1);
        bMtx1.setValueAt(testSize-1,i,1);
        bMtx1.setValueAt(i,0,1);
        bMtx1.setValueAt(i,testSize-1,1);
    }
    cout << bMtx1.toString() << endl;
    cout << bMtx1.dataToString() << endl;

    cout << "Toggle Linear Index:" << endl;
    BinaryMatrix bMtx2(testSize, testSize, BIT_ZERO);
    bool toggle = true;
    for(uint i=0; i<testSize*testSize; ++i) {
        bMtx2.setValueAt(i,toggle? BIT_ONE:BIT_ZERO);
        toggle = !toggle;
    }
    cout << bMtx2.dataToString() << endl;
}

void TestBinaryMatrix::testTransposeIdx() {
    cout << "----- TEST TRANSPOSE INDEX" << endl;
    uint testSize = 4;
    BinaryMatrix bMtx(testSize,testSize);

    for(uint i=0; i<testSize*testSize; ++i) {
        printf("%d ", i);
    }
    cout << endl;
    for(uint i=0; i<testSize*testSize; ++i) {
        printf("%d ", bMtx.transposeIndex(i));
    }
    cout << endl << endl;
}

void TestBinaryMatrix::testTranspose(){
    cout << "----- TEST TRANSPOSE" << endl;
    uint testSize = 3;
    BinaryMatrix bMtx = this->generateUpperDiag(testSize);

    cout << "U =" << endl;
    cout << bMtx.dataToString() << endl;
    bMtx.T();
    cout << "U.T = " << endl;
    cout << bMtx.dataToString() << endl;

    cout << "getDataAccessor transposed" << endl;
    for(uint i=0; i<bMtx.height(); ++i) {
        for(uint j=0; j<bMtx.width(); ++j) {
            std::pair<uint, uint> pos = bMtx.getDataAccessor(i, j);
            printf("[%d,%d]: ", pos.first, pos.second);
            printf("%u\n", bMtx.getBit(bMtx.data()[pos.first], pos.second));
        }
    }
}

void TestBinaryMatrix::testBinMultiply(){
    cout << "----- TEST MULTIPLY" << endl;
    uint testSize = 6;
    BinaryMatrix mtx0(testSize, testSize, BIT_ZERO);
    BinaryMatrix mtx1(testSize, testSize, BIT_ONE);

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
    uint testSize = 6;
    BinaryMatrix uDiag(testSize,testSize);
    BinaryMatrix lDiag(testSize,testSize);
    //Fill upper triangle
    for(uint i=0; i<testSize; ++i) {
        for(uint j=i; j < testSize; ++j) {
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
    BinaryMatrix zerosMtx(testSize, testSize);
    BinaryMatrix onesMtx(testSize, testSize, BIT_ONE);
    BinaryMatrix uDiag = this->generateUpperDiag(testSize);
    BinaryMatrix diag = this->generateDiag(testSize);

    mat dMtx(testSize, testSize);
    dMtx.fill(109.0);

    cout << "0 = " << endl;
    cout << zerosMtx.dataToString() << endl;
    cout << "0 * 109= " << endl;
    mat dResMtx = zerosMtx.doubleMultiply(dMtx);
    dResMtx.print();
    cout << endl;

    cout << "1 = " << endl;
    cout << onesMtx.dataToString() << endl;
    cout << "1 * 109= " << endl;
    dResMtx = onesMtx.doubleMultiply(dMtx);
    dResMtx.print();
    cout << endl;

    cout << "U = " << endl;
    cout << uDiag.dataToString() << endl;
    cout << "U * 109= " << endl;
    dResMtx = uDiag.doubleMultiply(dMtx);
    dResMtx.print();
    cout << endl;

    cout << "D = " << endl;
    cout << diag.dataToString() << endl;
    cout << "D * 109= " << endl;
    dResMtx = diag.doubleMultiply(dMtx);
    dResMtx.print();
    cout << endl;
}

bool TestBinaryMatrix::test_initWithArma_single(uint rows, uint cols) {
    // Initialize a random arma matrix
    arma::umat input2D = BinaryMatrix::randomArmaMat(rows, cols);

    // Initialize a BinaryMatrix
    BinaryMatrix bm(input2D);

    // Test values of binary matrix are equal to that of the arma::umat
    return bm.equalsArmaMat(input2D);
}

bool TestBinaryMatrix::test_initWithArma() {
    // Test with multiple inputs
    return test_initWithArma_single(3, 3)
        && test_initWithArma_single(50, 50)
        && test_initWithArma_single(30, 101)
        && test_initWithArma_single(57, 98);
}

bool TestBinaryMatrix::test_im2col_single(uint rows, uint cols, uint block_width, uint block_height,
                                   uint padding, uint stride) {
    // Generate a random binary matrix
    arma::umat input2D = BinaryMatrix::randomArmaMat(rows, cols);
    BinaryMatrix bm(input2D);

    // Compare im2col result for binary matrix and arma
    BinaryMatrix bmResult = bm.im2col(block_width, block_height, padding, stride);
    arma::umat armaResult = BinaryMatrix::im2colArmaMat(input2D, block_width, block_height, padding, stride);

    return bmResult.equalsArmaMat(armaResult);
}

bool TestBinaryMatrix::test_im2col() {
    // Tests with different inputs
    return test_im2col_single();
    /*
        && test_im2col_single(3, 3)
        && test_im2col_single(3, 3, 3, 3, 1, 1)
        && test_im2col_single(3, 3, 3, 3, 1, 2)
        && test_im2col_single(5, 5, 3, 3, 1, 2)
        && test_im2col_single(7, 9, 5, 5, 2, 1)
        && test_im2col_single(8, 6, 3, 3, 1, 2);
    */
}

bool TestBinaryMatrix::runAllTests(){
    /*
    testCreateAndPrint();
    testGetBit();
    testSetBit();
    testTransposeIdx();
    testTranspose();
    testBinMultiply();
    testTBinMultiply();
    testDoubleMultiply();
    */
    return test_initWithArma() && test_im2col();
}