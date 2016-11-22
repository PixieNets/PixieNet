//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <utility>
#include <cstdio>
#include "BinaryMatrix.h"

BinaryMatrix::BinaryMatrix(int w, int h) {
    this->init(w, h, 0);
}

/**
 * Initializes a 2D binary matrix
 * @param w - width of matrix
 * @param h - height of matrix
 * @return (none)
 */
BinaryMatrix::BinaryMatrix(int w, int h, int initVal) {
    this->init(w, h, initVal);
}

/**
 * Destructor for binary matrix, delete the data
 */
BinaryMatrix::~BinaryMatrix() {
    if(this->data != nullptr)
        delete[] this->data;
}

void BinaryMatrix::init(int w, int h, int initVal) {
    this->width = w;
    this->height = h;
    this->baseSize = sizeof(uchar) * 8;
    this->transposed = false;

    //Initialize data, data is stored in a linear form
    int n = w * h;
    this->dataLength = n / baseSize;
    if (n % baseSize != 0)
        ++this->dataLength;
    this->data = new uchar[dataLength];
    uchar val = (initVal == 0)? BIT_ZERO:BIT_ONE;
    for(int i = 0; i < this->dataLength; ++i) {
        this->data[i] = val;
    }
}

/**
 * Toggles the "transposed" switch for binary matrix
 */
void BinaryMatrix::T() {
    this->transposed = !this->transposed;
}

int BinaryMatrix::transposeIndex(int idx) {
    return (idx / this->width + ((idx % this->width) * this->width));
}

IntPair BinaryMatrix::getDataAccessor(int row, int col) {
    int idx = (this->transposed)? (col * this->width + row):(row * this->width + col);
    return std::make_pair(idx / this->baseSize, idx % this->baseSize);
}

int BinaryMatrix::getLinearIndex(int row, int col, int height, int width, bool transposed) {
    return (transposed)? (col * width + row):(row * width + col);
}

/**
 * Generates the (row, col) position for the i^{th} bit stored in the binary matrix
 * and ensures that access takes into account if the matrix is transposed
 * @param i - bit index in matrix
 * @param rows - #rows in binary matrix
 * @param cols - #cols in binary matrix
 * @param transposed - boolean to test if the matrix is tranposed and hence read differently
 * @return - (dataPos, bitIndex) position of the bit in the array
 */
IntPair BinaryMatrix::elemAccessor(int i, int rows, int cols, bool transposed) {
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
uchar BinaryMatrix::getBit(uchar elem, int bit_id) {
    return ((elem >> (this->baseSize-1 - bit_id)) & 1) ? BIT_ONE:BIT_ZERO;
}

/**
 * Sets the bit at a location in a binary matrix
 * @param elem - the row containing bit to be modified
 * @param bit_id - index of the bit to be modified
 * @param bitValue - the new bit value
 * @return - the row with the new modified bit
 */
uchar BinaryMatrix::setBit(uchar elem, int bit_id, uchar bitValue) {
    uchar mask = (uchar) (1 << (this->baseSize-1 - bit_id));
    if (bitValue == 0) {
        return (elem & !mask);
    } else {
        return (elem | mask);
    }
}

uchar BinaryMatrix::getValueAt(int idx) {
    assert(idx < this->height*this->width);

    if(this->transposed)    idx = transposeIndex(idx);
    IntPair pos = std::make_pair(idx / this->baseSize, idx % this->baseSize);
    return this->getBit(this->data[pos.first], pos.second);
}

uchar BinaryMatrix::getValueAt(int row, int col) {
    assert( row < this->height);
    assert( col < this->width);

    //IntPair pos = elemAccessor(row*this->width+col, this->dataLength, this->baseSize, this->transposed);
    IntPair pos = this->getDataAccessor(row, col);
    return this->getBit(this->data[pos.first], pos.second);
}

void BinaryMatrix::setValueAt(int idx, uchar bitValue) {
    assert(idx < this->height*this->width);

    if(this->transposed)    idx = transposeIndex(idx);
    IntPair pos = std::make_pair(idx / this->baseSize, idx % this->baseSize);
    this->setBit(this->data[pos.first], pos.second, bitValue);
}

void BinaryMatrix::setValueAt(int row, int col, uchar bitValue) {
    assert( row < this->height);
    assert( col < this->width);

    //IntPair pos = this->elemAccessor( (row*this->width)+col, this->dataLength, this->baseSize, this->transposed);
    IntPair pos = this->getDataAccessor(row, col);
    this->data[pos.first] = this->setBit(this->data[pos.first], pos.second, bitValue);
}

/**
 * Multiplies two binary matrices. Binary matrix multiplciation is the
 * Haddamard product of the matrices, different from regular matrix multiplication.
 * @param other - The binary matrix we multiply current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
BinaryMatrix BinaryMatrix::binMultiply(const BinaryMatrix &other) {
    BinaryMatrix res(this->width, this->height);
    for(int i = 0; i < this->dataLength; ++i) {
        res.data[i] = ~(this->data[i] ^ other.data[i]);
    }
    return res;
}

/**
 * Multiplies two matrices, one of which is transposed, bit-wise, again to
 * return their Hadamhard product.
 * @param other - the matrix to multiply the current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
BinaryMatrix BinaryMatrix::tBinMultiply(const BinaryMatrix &other) {
    int w = this->width;
    int h = this->height;
    if (this->transposed) {
        w = other.width;
        h = other.height;
    }
    BinaryMatrix res(w, h);
    int this_n = this->dataLength;
    int other_n = other.dataLength;
    IntPair this_rc;
    IntPair other_rc;
    IntPair res_rc;
    uchar   answer_c;
    int     thisIdx, otherIdx;
    for (int row = 0; row < this->height; ++row) {
        for(int col = 0; col < this->width; ++col) {
            thisIdx = getLinearIndex(row, col, this->height, this->width, this->transposed);
            otherIdx = getLinearIndex(row, col, other.height, other.width, other.transposed);
            this_rc = this->elemAccessor(thisIdx, this_n, this->baseSize, false);
            other_rc = this->elemAccessor(otherIdx, other_n, other.baseSize, false);
            res_rc = this->transposed? other_rc : this_rc;

            answer_c = (uchar) (~(getBit(this->data[this_rc.first], this_rc.second)
                                  ^ getBit(other.data[other_rc.first], other_rc.second)) & 1);
            res.data[res_rc.first] = setBit(res.data[res_rc.first], res_rc.second, answer_c);
        }
    }
    return res;
}

/*
 * The operations are done row-wise
 * !!! Watch out and remember to clean the memory
 */

double* BinaryMatrix::doubleMultiply(const double *other) {
    double* res = new double[this->dataLength];

    IntPair linearPos;
    for(int row=0; row < this->height; ++row) {
        for(int col=0; col < this->width; ++col) {
            if( this->getValueAt(row, col) == 0)
                res[row*col] = other[row*col]*-1;
            else
                res[row*col] = other[row*col];
        }
    }

    return res;
}

int BinaryMatrix::bitCount() {
    int count = 0;
    for(int i=0; i<this->dataLength; ++i) {
        for(int b=0; b<this->baseSize; ++b) {
            if(this->data[i]>>b & 1)
                count++;
        }
    }
    return count;
}

void BinaryMatrix::print() {
    for(int row = 0; row < this->height; ++row) {
        for(int col = 0; col < this->width; ++col) {
            printf("%u ", getValueAt(row, col));
        }
        printf("\n");
    }
}

std::string BinaryMatrix::toString() {
    std::string transStr = (this->transposed) ? "Yes":"No";
    std::string res = "< Binary Matrix rows:" + std::to_string(this->height)
                        + " cols:" + std::to_string(this->width)
                        + " transposed:" + transStr + " >";
    return res;
}

std::string BinaryMatrix::dataToString() {
    std::string res;

    for(int row=0; row < this->height; ++row) {
        for(int col=0; col < this->width; ++col) {
            res += std::to_string(getValueAt(row, col)) + " ";
        }
        res += "\n";
    }
    return res;
}

/**
 * Defines the multiplication operator for 2 binary matrices as their
 * Hadamhard product
 * @param other - the matrix to multiply the current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
BinaryMatrix BinaryMatrix::operator*(const BinaryMatrix &other ) {
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
