//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <utility>
#include <cstdio>
#include "BinaryMatrix.h"

#define intPair std::pair<int, int>

/**
 * Initializes a 2D binary matrix
 * @param w - width of matrix
 * @param h - height of matrix
 * @return (none)
 */
void BinaryMatrix::BinaryMatrix(int w, int h) {
    this->width = w;
    this->height = h;
    this->baseSize = sizeof(char) * 8;
    this->transposed = false;

    //Initialize data, data is stored in a linear form
    int n = w*h;
    this->dataLength = (n % baseSize == 0)? n/baseSize : n/baseSize +1;
    this->data = new char[dataLength];
    for(int i=0; i<this->dataLength; ++i) {
        this->data[i]=0;
    }
}

/**
 * Destructor for binary matrix, delete the data
 */
void BinaryMatrix::~BinaryMatrix() {
    if(data != NULL)    delete[] data;
}

/**
 * Toggles the "transposed" switch for binary matrix
 */
void BinaryMatrix::T() {
    this->transposed = !this->transposed;
}

/**
 * Multiplies two binary matrices. Binary matrix multiplciation is the
 * Haddamard product of the matrices, different from regular matrix multiplication.
 * @param other - The binary matrix we multiply current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
BinaryMatrix BinaryMatrix::binMultiply(const BinaryMatrix& other) {
    BinaryMatrix res(this->width, this->height);
    for(int i = 0; i < this->dataLength; ++i) {
        res.data[i] = !(this->data[i] ^ other.data[i]);
    }
    return res;
}

/**
 * Generates the (row, col) position for the i^{th} bit stored in the binary matrix
 * and ensures that access takes into account if the matrix is transposed
 * @param i - bit index in matrix
 * @param rows - #rows in binary matrix
 * @param cols - #cols in binary matrix
 * @param transposed - boolean to test if the matrix is tranposed and hence read differently
 * @return - (row, col) position of the bit in the matrix
 */
std::pair<int, int> BinaryMatrix::elem_accessor(int i, int rows, int cols, bool transposed) {
    if (transposed) {
        return std::make_pair(i % rows, i / rows);
    } else {
        return std::make_pair(i / cols, i % cols);
    }
}

/**
 * Returns the bit at a location in the binary matrix
 * @param elem - the row containing the bit
 * @param bit_id - the index of the bit in the row
 * @return - the bit stored at 'bit_id'
 */
char BinaryMatrix::get_bit(char elem, int bit_id) {
    return (elem >> (this->baseSize - bit_id)) & 1;
}

/**
 * Sets the bit at a location in a binary matrix
 * @param elem - the row containing bit to be modified
 * @param bit_id - index of the bit to be modified
 * @param bit - the new bit value
 * @return - the row with the new modified bit
 */
char BinaryMatrix::set_bit(char elem, int bit_id, char bit) {
    char mask = 1 << (this->baseSize - bit_id);
    if (bit == 0) {
        return (elem & !mask);
    } else {
        return (elem | mask);
    }
}

char BinaryMatrix::getValueAt(int i) {
    assert(i < this->width*this->height);
    intPair pos = elem_accessor(i, this->dataLength, this->baseSize, this->transposed);
    return this->get_bit(this->data[pos.first], pos.second);
}

/**
 * Multiplies two matrices, one of which is transposed, bit-wise, again to
 * return their Hadamhard product.
 * @param other - the matrix to multiply the current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
BinaryMatrix BinaryMatrix::tBinMultiply(const BinaryMatrix& other) {
    int w = this->width;
    int h = this->height;
    if (this->transposed) {
        w = other.width;
        h = other.height;
    }
    BinaryMatrix res(w, h);
    int this_n = this->dataLength;
    int other_n = other.dataLength;
    for (int bit_id = 0; bit_id < (w * h); ++bit_id) {
        std::pair<int, int> this_rc = elem_accessor(bit_id, this_n, this->baseSize, this->transposed);
        std::pair<int, int> other_rc = elem_accessor(bit_id, other_n, other.baseSize, other.transposed);
        std::pair<int, int> res_rc = this->transposed? other_rc : this_rc;
        char this_c = this->data[this_rc.first];
        char other_c = other.data[other_rc.first];
        char res_c = res.data[res_rc.first];

        char answer = !(get_bit(this_c, this_rc.second) ^ get_bit(other_c, other_rc.second)) & 1;
        res.data[res_rc.first] = set_bit(res_c, res_rc.second, answer);
    }
    return res;
}

/*
 * The operations are done row-wise
 * !!! Watch out and remember to clean the memory
 */

double* BinaryMatrix::doubleMultiply(const double* other) {
    double* res = new double[this->dataLength];

    intPair linearPos;
    for(int row=0; row < this->height; ++row) {
        for(int col=0; col < this->width; ++col) {
            linearPos = elem_accessor(row*col, this->dataLength, this->baseSize, this->transposed);
            if( get_bit(this->data[linearPos.first], linearPos.second) == 0) {
                res[row*col] = other[row*col]*-1;
            }
            else {
                res[row*col] = other[row*col];
            }
        }
    }

    return res;
}

int BinaryMatrix::bitCount() {
    int count = 0;
    for(int i=0; i<this->dataLength; ++i) {
        for(int b=0; b<this->baseSize; ++b) {
            if(this->data[i]>>b&1)
                count++;
        }
    }
    return count;
}

void BinaryMatrix::print() {
    for(int row; row < this->height; ++row) {
        for(int col; col < this->width; ++col) {
            printf("%u ",getValueAt(row*col));
        }
        printf("\n");
    }
}

/**
 * Defines the multiplication operator for 2 binary matrices as their
 * Hadamhard product
 * @param other - the matrix to multiply the current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
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
