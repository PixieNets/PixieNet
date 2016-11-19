//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include "BinaryMatrix.h"

void BinaryMatrix::BinaryMatrix(int w, int h) {
    this->width = w;
    this->height = h;
    baseSize = sizeof(char);

    int n = w*h;
    dataLength = (n % baseSize == 0)? n/baseSize : n/baseSize +1;
    this->data = new char[dataLength];
}

void BinaryMatrix::~BinaryMatrix() {
    delete[] data;
}

void BinaryMatrix::T() {
    this->transposed = !this->transposed;
}

BinaryMatrix BinaryMatrix::binMultiply(const BinaryMatrix& other) {
    BinaryMatrix res(this->width, this->height);
    for(int i = 0; i < this->dataLength; ++i) {
        res.data[i] = !(this->data[i] ^ other.data[i]);
    }
    return res;
}

BinaryMatrix BinaryMatrix::tBinMultiply(const BinaryMatrix& other) {

}

double BinaryMatrix::doubleMultiply(const double& other) {

}

BinaryMatrix BinaryMatrix::operator*(const BinaryMatrix& other ) {
    assert(this->width == other.width);
    assert(this->height == other.height)
    if(this->transposed != other.transposed) {
        return this->tBinMultiply(other);
    }
    else {
        return this->binMultiply(other);
    }
}
