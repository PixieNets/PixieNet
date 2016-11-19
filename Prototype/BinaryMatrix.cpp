//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <utility>
#include "BinaryMatrix.h"

#define intPair std::pair<int, int>

void BinaryMatrix::BinaryMatrix(int w, int h) {
    this->width = w;
    this->height = h;
    this->baseSize = sizeof(char) * 8;

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

// Binary matrix multiplciation is the Haddamard product of the matrices
// It is different from the regular matrix mutltiplication
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

char get_bit(char elem, int bit_id) {
    return (elem >> (this->baseSize - bit_id)) & 1;
}

char set_bit(char elem, int bit_id, char bit) {
    // assumes originally the bit is set to 0
    char mask = 1 << ( - bit_id);
    char new_elem = (elem | (bit << mask));
    return new_elem;
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
    int res_n = res.dataLength;
    BinaryMatrix res(w, h);
    for (int bit_id = 0; bit_id < (w * h); ++bit_id) {
<<<<<<< HEAD
        std::pair<int, int> this_rc = elem_accessor(bit_id, this_n, 8, this->transposed);
        std::pair<int, int> other_rc = elem_accessor(bit_id, other_n, 8, other.transposed);
        std::pair<int, int> res_rc = elem_accessor(bit_Id, res.dataLength, 8, res.transposed);
=======
        std::pair<int, int> this_rc = elem_accessor(bit_id, this_n, this->baseSize, this->transposed);
        std::pair<int, int> other_rc = elem_accessor(bit_id, other_n, this->baseSize, other.transposed);
        std::pair<int, int> res_rc = elem_accessor(bit_Id, res_n, this->baseSize, res.transposed);
>>>>>>> 06de3fd37cd1b59a3ce43694cd1792aad11cf0b1
        char this_c = this->data[this_rc.first];
        char other_c = other.data[other_rc.first];
        char res_c = res.data[res_rc.first];

        char answer = !(get_bit(this_c, this_rc.second) ^ get_bit(other_c, other_rc.second)) & 1;
        res.data[res_rc.first] = set_bit(res_c, res_rc.second, answer);
    }
    return res;
}

//The operations are done row-wise
double* BinaryMatrix::doubleMultiply(const double* other) {

    double* res = new double[this->height][this->width];
    for(int row=0; row < this->height; ++row) {
        for(int col=0; col < this->width; ++col) {

        }
    }

    return res;
}

// Remember it's 0-based
char BinaryMatrix::getBitPos(int row, int col) {
    int i = (row+1) * (col+1);
    intPair pos = elem_accessor(i, this->dataLength, this->baseSize, this->transposed);
    return (this->data[pos.first]>>pos.second & 1);
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
