//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <utility>
#include <cstdio>
#include "BinaryMatrix.h"

BinaryMatrix::BinaryMatrix(uint w, uint h) {
    this->init(w, h, 0);
}

/**
 * Initializes a 2D binary matrix
 * @param w - width of matrix
 * @param h - height of matrix
 * @return (none)
 */
BinaryMatrix::BinaryMatrix(uint w, uint h, uint8 initVal) {
    this->init(w, h, initVal);
}

/**
 * Destructor for binary matrix, delete the data
 */
BinaryMatrix::~BinaryMatrix() {
    if(this->bm_data != nullptr)
        delete[] this->bm_data;
}

void BinaryMatrix::init(uint w, uint h, uint8 initVal) {
    this->bm_width = w;
    this->bm_height = h;
    this->bm_baseBitSize = sizeof(uint8) * 8;
    this->bm_transposed = false;

    //Initialize data, data is stored in a linear form
    int n = w * h;
    this->bm_dataLength = n / bm_baseBitSize;
    if (n % bm_baseBitSize != 0)
        ++this->bm_dataLength;
    this->bm_data = new uint8[this->bm_dataLength];
    uint8 val = (uint8) ((initVal == BIT_ZERO) ? 0 : ~0);  //8 zeroes or 8 ones
    for(int i = 0; i < this->bm_dataLength; ++i) {
        this->bm_data[i] = val;
    }
}

/**
 * Toggles the "transposed" switch for binary matrix
 */
void BinaryMatrix::T() {
    this->bm_transposed = !this->bm_transposed;
}

uint BinaryMatrix::transposeIndex(uint idx) {
    return (idx / this->bm_width + ((idx % this->bm_width) * this->bm_width));
}

uint BinaryMatrix::transposeIndex(uint idx, uint width) {
    return (idx / width + ((idx % width) * width));
}

uIntPair BinaryMatrix::getDataAccessor(uint row, uint col) {
    uint idx = (this->bm_transposed)? (col * this->bm_width + row): (row * this->bm_width + col);
    return std::make_pair((idx / this->bm_baseBitSize), (idx % this->bm_baseBitSize));
}

uint BinaryMatrix::getLinearIndex(uint row, uint col, uint height, uint width, bool transposed) {
    return (transposed)? (col * width + row): (row * width + col);
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
uIntPair BinaryMatrix::elemAccessor(uint i, uint rows, uint cols, bool transposed) {
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
uint8 BinaryMatrix::getBit(uint8 elem, uint bit_id) {
    return ((elem >> (this->bm_baseBitSize-1 - bit_id)) & 1) ? BIT_ONE:BIT_ZERO;
}

/**
 * Sets the bit at a location in a binary matrix
 * @param elem - the row containing bit to be modified
 * @param bit_id - index of the bit to be modified
 * @param bitValue - the new bit value
 * @return - the row with the new modified bit
 */
uint8 BinaryMatrix::setBit(uint8 elem, uint bit_id, uint8 bitValue) {
    uint8 mask = (uint8) (1 << (this->bm_baseBitSize-1 - bit_id));
    if (bitValue == 0) {
        return (elem & !mask);
    } else {
        return (elem | mask);
    }
}

uint8 BinaryMatrix::getValueAt(uint idx) {
    assert(idx < this->bm_height*this->bm_width);

    if(this->bm_transposed)
        idx = transposeIndex(idx);
    uIntPair pos = std::make_pair(idx / this->bm_baseBitSize, idx % this->bm_baseBitSize);
    return this->getBit(this->bm_data[pos.first], pos.second);
}

uint8 BinaryMatrix::getValueAt(uint row, uint col) {
    assert( row < this->bm_height);
    assert( col < this->bm_width);

    //IntPair pos = elemAccessor(row*this->width+col, this->dataLength, this->baseBitSize, this->transposed);
    uIntPair pos = this->getDataAccessor(row, col);
    return this->getBit(this->bm_data[pos.first], pos.second);
}

void BinaryMatrix::setValueAt(uint idx, uint8 bitValue) {
    assert(idx < this->bm_height*this->bm_width);

    if(this->bm_transposed)
        idx = transposeIndex(idx);
    uIntPair pos = std::make_pair(idx / this->bm_baseBitSize, idx % this->bm_baseBitSize);
    this->setBit(this->bm_data[pos.first], pos.second, bitValue);
}

void BinaryMatrix::setValueAt(uint row, uint col, uint8 bitValue) {
    assert( row < this->bm_height);
    assert( col < this->bm_width);

    //IntPair pos = this->elemAccessor( (row*this->width)+col, this->dataLength, this->baseBitSize, this->transposed);
    uIntPair pos = this->getDataAccessor(row, col);
    this->bm_data[pos.first] = this->setBit(this->bm_data[pos.first], pos.second, bitValue);
}

/**
 * Multiplies two binary matrices. Binary matrix multiplciation is the
 * Haddamard product of the matrices, different from regular matrix multiplication.
 * @param other - The binary matrix we multiply current matrix with
 * @return - BinaryMatrix containing the same number of total bits as the 2 input matrices
 */
BinaryMatrix BinaryMatrix::binMultiply(const BinaryMatrix &other) {
    assert(this->bm_transposed == other.bm_transposed);

    BinaryMatrix res(this->bm_width, this->bm_height);
    for(uint i = 0; i < this->bm_dataLength; ++i) {
        res.bm_data[i] = ~(this->bm_data[i] ^ other.bm_data[i]);
    }
    res.bm_transposed = this->bm_transposed;
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
    if( (this->bm_transposed && !other.bm_transposed) ||
        (!this->bm_transposed && other.bm_transposed) ) {
        assert(this->bm_width == other.bm_height);
        assert(this->bm_height == other.bm_width);
    }
    else {
        assert(this->bm_height == other.bm_height);
        assert(this->bm_width == other.bm_width);
    }

    // Since this is the haddamard by design we choose to keep the
    // dimensions of the non-transposed matrix, since by transposing
    // we are trying to align the matrix to be multiplied
    int resW = this->bm_width;
    int resH = this->bm_height;
    if (this->bm_transposed) {
        resW = other.bm_width;
        resH = other.bm_height;
    }
    BinaryMatrix res(resW, resH);

    uIntPair this_rc, other_rc, res_rc;
    uint8    answer_c;
    uint     thisIdx, otherIdx;

    uint numBits = this->bm_height * this->bm_width;
    for(uint i = 0; i < numBits; ++i) {
        thisIdx = (this->bm_transposed)? transposeIndex(i, this->bm_width) : i;
        otherIdx = (other.bm_transposed)? transposeIndex(i, other.bm_width) : i;

        this_rc = this->elemAccessor(thisIdx, this->bm_dataLength, this->bm_baseBitSize, false);
        other_rc = this->elemAccessor(otherIdx, other.bm_dataLength, other.bm_baseBitSize, false);
        res_rc = this->bm_transposed? other_rc : this_rc;

        answer_c = (uint8) (~(getBit(this->bm_data[this_rc.first], this_rc.second)
                              ^ getBit(other.bm_data[other_rc.first], other_rc.second)) & 1);
        res.bm_data[res_rc.first] = setBit(res.bm_data[res_rc.first], res_rc.second, answer_c);
    }

    return res;
}

/**
 *
 **/

mat BinaryMatrix::doubleMultiply(const mat &other) {
    mat res(this->bm_height, this->bm_width);
    uIntPair linearPos;

    for(uint row = 0; row < this->bm_height; ++row) {
        for(uint col = 0; col < this->bm_width; ++col) {
            if( this->getValueAt(row,col) == BIT_ZERO)
                res(row, col) = other(row,col) * -1;
            else
                res(row, col) = other(row,col);
        }
    }

    return res;
}

uint BinaryMatrix::bitCount() {
    uint count = 0;
    for(uint i = 0; i < this->bm_dataLength; ++i) {
        for(uint b = 0; b < this->bm_baseBitSize; ++b) {
            if(this->bm_data[i]>>b & 1)
                ++count;
        }
    }
    return count;
}

void BinaryMatrix::print() {
    for(uint row = 0; row < this->bm_height; ++row) {
        for(uint col = 0; col < this->bm_width; ++col) {
            printf("%u ", getValueAt(row, col));
        }
        printf("\n");
    }
}

std::string BinaryMatrix::toString() {
    std::string transStr = (this->bm_transposed) ? "Yes":"No";
    std::string res = "< BinaryMatrix rows:" + std::to_string(this->bm_height)
                        + " cols:" + std::to_string(this->bm_width)
                        + " transposed:" + transStr + " >";
    return res;
}

std::string BinaryMatrix::dataToString() {
    std::string res;

    for(uint row = 0; row < this->bm_height; ++row) {
        for(uint col = 0; col < this->bm_width; ++col) {
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
    if(this->bm_transposed != other.bm_transposed) {
        assert(this->bm_width == other.bm_height);
        assert(this->bm_height == other.bm_width);
        return this->tBinMultiply(other);
    }
    else {
        assert(this->bm_width == other.bm_width);
        assert(this->bm_height == other.bm_height);
        return this->binMultiply(other);
    }
}


BinaryMatrix BinaryMatrix::im2col(BinaryMatrix &input, uint block_width, uint block_height,
                                  uint padding, uint stride) {
    uint n = block_width * block_height;
    uint rows_out = (input.bm_height - block_height + 2 * padding) / stride + 1;
    uint cols_out = (input.bm_width - block_width + 2 * padding) / stride + 1;
    BinaryMatrix result(rows_out * cols_out, n);

    uint res_row = 0;
    for (uint row = padding; row < (input.bm_height - padding - 1); ++row) {
        for (uint col = padding; col < (input.bm_width - padding - 1); ++col) {
            uint res_col = 0;
            for (uint srow = row - padding; srow < (row + padding); ++srow) {
                for (uint scol = col - padding; scol < (col + padding); ++scol) {
                    // In general
                    // result[res_row, res_col++] = input[srow, scol];
                    result.setValueAt(res_row, res_col, input.getValueAt(srow, scol));
                    ++res_col;
                }
            }
            ++res_row;
        }
    }
    // Check that we capture as many blocks as we had to
    assert(res_row == (rows_out * cols_out));

    return result;
}
