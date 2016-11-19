//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <utility>
#include "BinaryMatrix.h"

void BinaryMatrix::BinaryMatrix(int w, int h) {
    this->width = w;
    this->height = h;
    baseSize = sizeof(char);

    int n = w*h;
    this->dataLength = (n % baseSize == 0)? n/baseSize : n/baseSize +1;
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

std::pair<int, int> elem_accessor(int i, int rows, int cols, bool transposed) {
    if (transposed) {
        return std::make_pair(i % rows, i / rows);
    } else {
        return std::make_pair(i / cols, i % cols);
    }
}

BinaryMatrix BinaryMatrix::tBinMultiply(const BinaryMatrix& other) {
    int w = this->width;
    int h = this->height;
    if (this->transposed) {
        w = other.width;
        h = other.height;
    }
    int this_n = this->dataLength;
    int other_n = other.dataLength;
    BinaryMatrix res(w, h);
    int this_bit_id = 0, other_bit_id = 0;
    for (int bit_id = 0; bit_id < (w * h); ++bit_id) {
        this_bit_id = elem_accessor(bit_id, 8, this_n, this->transposed);
        other_bit_id = elem_accessor(bit_id, 8, other_n, other.transposed);
        res.data[bit_id] = this->data[this_bit_id] * other.data[other_bit_id];
    }
    return res;
}

double BinaryMatrix::doubleMultiply(const double& other) {

}

BinaryMatrix BinaryMatrix::operator*(const BinaryMatrix& other ) {
    if(this->transposed != other.transposed) {
        assert(this->width == other.height);
        assert(this->height == other.width);
        return this->tBinMultiply(other);
    }
    else {
        assert(this->width == other.width);
        assert(this->height == other.height);
        return this->binMultiply(other);
    }
}
