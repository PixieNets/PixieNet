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
BinaryMatrix::BinaryMatrix(int w, int h, uint8 initVal) {
    this->init(w, h, initVal);
}

/**
 * Destructor for binary matrix, delete the data
 */
BinaryMatrix::~BinaryMatrix() {
    if(this->data != nullptr)
        delete[] this->data;
}

void BinaryMatrix::init(int w, int h, uint8 initVal) {
    this->width = w;
    this->height = h;
    this->baseBitSize = sizeof(uint8) * 8;
    this->transposed = false;

    //Initialize data, data is stored in a linear form
    int n = w * h;
    this->dataLength = n / baseBitSize;
    if (n % baseBitSize != 0)
        ++this->dataLength;
    this->data = new uint8[dataLength];
    uint8 val = (initVal == BIT_ZERO)? 0 : ~0;  //8 zeroes or 8 ones
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

int BinaryMatrix::transposeIndex(int idx, int width) {
    return (idx / width + ((idx % width) * width));
}

IntPair BinaryMatrix::getDataAccessor(int row, int col) {
    int idx = (this->transposed)? (col * this->width + row):(row * this->width + col);
    return std::make_pair(idx / this->baseBitSize, idx % this->baseBitSize);
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
uint8 BinaryMatrix::getBit(uint8 elem, int bit_id) {
    return ((elem >> (this->baseBitSize-1 - bit_id)) & 1) ? BIT_ONE:BIT_ZERO;
}

/**
 * Sets the bit at a location in a binary matrix
 * @param elem - the row containing bit to be modified
 * @param bit_id - index of the bit to be modified
 * @param bitValue - the new bit value
 * @return - the row with the new modified bit
 */
uint8 BinaryMatrix::setBit(uint8 elem, int bit_id, uint8 bitValue) {
    uint8 mask = (uint8) (1 << (this->baseBitSize-1 - bit_id));
    if (bitValue == 0) {
        return (elem & !mask);
    } else {
        return (elem | mask);
    }
}

uint8 BinaryMatrix::getValueAt(int idx) {
    assert(idx < this->height*this->width);

    if(this->transposed)    idx = transposeIndex(idx);
    IntPair pos = std::make_pair(idx / this->baseBitSize, idx % this->baseBitSize);
    return this->getBit(this->data[pos.first], pos.second);
}

uint8 BinaryMatrix::getValueAt(int row, int col) {
    assert( row < this->height);
    assert( col < this->width);

    //IntPair pos = elemAccessor(row*this->width+col, this->dataLength, this->baseBitSize, this->transposed);
    IntPair pos = this->getDataAccessor(row, col);
    return this->getBit(this->data[pos.first], pos.second);
}

void BinaryMatrix::setValueAt(int idx, uint8 bitValue) {
    assert(idx < this->height*this->width);

    if(this->transposed)    idx = transposeIndex(idx);
    IntPair pos = std::make_pair(idx / this->baseBitSize, idx % this->baseBitSize);
    this->setBit(this->data[pos.first], pos.second, bitValue);
}

void BinaryMatrix::setValueAt(int row, int col, uint8 bitValue) {
    assert( row < this->height);
    assert( col < this->width);

    //IntPair pos = this->elemAccessor( (row*this->width)+col, this->dataLength, this->baseBitSize, this->transposed);
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
    assert(this->transposed == other.transposed);

    BinaryMatrix res(this->width, this->height);
    for(int i = 0; i < this->dataLength; ++i) {
        res.data[i] = ~(this->data[i] ^ other.data[i]);
    }
    res.transposed = this->transposed;
    return res;
}

/**
 * Multiplies two matrices, one of which is transposed, bit-wise, again to
 * return their Hadamhard product. The resulting matrix will have the size of the non-transposed.
 * @param other - the matrix to multiply the current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
BinaryMatrix BinaryMatrix::tBinMultiply(const BinaryMatrix &other) {
    // Verify that dimensions correspond
    if( (this->transposed && !other.transposed) ||
        (!this->transposed && other.transposed) ) {
        assert(this->width == other.height);
        assert(this->height == other.width);
    }
    else {
        assert(this->height == other.height);
        assert(this->width == other.width);
    }

    // Since this is the haddamard by design we choose to keep the
    // dimensions of the non-transposed matrix, since by transposing
    // we are trying to align the matrix to be multiplied
    int resW = this->width;
    int resH = this->height;
    if (this->transposed) {
        resW = other.width;
        resH = other.height;
    }
    BinaryMatrix res(resW, resH);

    IntPair this_rc, other_rc, res_rc;
    uint8   answer_c;
    int     thisIdx, otherIdx;

    int numBits = this->height * this->width;
    for(int i=0; i < numBits; ++i) {
        thisIdx = (this->transposed)? transposeIndex(i, this->width) : i;
        otherIdx = (other.transposed)? transposeIndex(i, other.width) : i;

        this_rc = this->elemAccessor(thisIdx, this->dataLength, this->baseBitSize, false);
        other_rc = this->elemAccessor(otherIdx, other.dataLength, other.baseBitSize, false);
        res_rc = this->transposed? other_rc : this_rc;

        answer_c = (uint8) (~(getBit(this->data[this_rc.first], this_rc.second)
                              ^ getBit(other.data[other_rc.first], other_rc.second)) & 1);
        res.data[res_rc.first] = setBit(res.data[res_rc.first], res_rc.second, answer_c);
    }

    return res;
}

/**
 *
 **/

mat BinaryMatrix::doubleMultiply(const mat &other) {
    mat res(this->height, this->width);
    IntPair linearPos;

    for(int row=0; row < this->height; ++row) {
        for(int col=0; col < this->width; ++col) {
            if( this->getValueAt(row,col) == BIT_ZERO)
                res(row, col) = other(row,col)*-1;
            else
                res(row, col) = other(row,col);
        }
    }

    return res;
}

int BinaryMatrix::bitCount() {
    int count = 0;
    for(int i=0; i<this->dataLength; ++i) {
        for(int b=0; b<this->baseBitSize; ++b) {
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
    std::string res = "< BinaryMatrix rows:" + std::to_string(this->height)
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
