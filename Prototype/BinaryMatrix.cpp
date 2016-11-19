//
// Created by Zal on 11/19/16.
//

#include "BinaryMatrix.h"

void BinaryMatrix::BinaryMatrix(int w, int h) {
    this->width = w;
    this->height = h;
    baseSize = sizeof(char);

    int n = w*h;
    int totalBases = (n % baseSize == 0)? n/baseSize : n/baseSize +1;
    this->data = new char[totalBases];
}

void BinaryMatrix::~BinaryMatrix() {
    delete[] data;
}

void BinaryMatrix::T() {
    this->transposed = !this->transposed;
}

BinaryMatrix BinaryMatrix::binMultiply(const BinaryMatrix& other) {

}

BinaryMatrix BinaryMatrix::tBinMultiply(const BinaryMatrix& other) {

}

BinaryMatrix BinaryMatrix::operator*(const BinaryMatrix& other ) {
    if(this->transposed != other.transposed) {
        return this->tBinMultiply(other);
    }
    else {
        return this->binMultiply(other);
    }
}
