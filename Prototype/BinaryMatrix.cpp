//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <utility>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "BinaryMatrix.h"


/**
 * Initializes a 2D binary matrix
 * @param w - width of matrix
 * @param h - height of matrix
 * @return (none)
 */
BinaryMatrix::BinaryMatrix(uint w, uint h, uint8 initVal) {
    this->init(w, h, initVal);
}

BinaryMatrix::BinaryMatrix(uint w, uint h, bool randomized, uint n) {
    this->init(w, h, 0);
    if (randomized) {
        // randomly set some bits to 1
        std::vector<uint> indices = this->randIndices(this->bm_dataLength, n);
        for (uint idx : indices) {
            setValueAt(idx, BIT_ONE);
        }
    }
}

BinaryMatrix::BinaryMatrix(arma::umat input2D) {
    if (input2D.is_empty()) {
        throw std::invalid_argument("[BinaryMatrix::constructor] arma::mat input2D (arg1) should be a non-empty matrix\n");
    }

    this->init((uint) input2D.n_cols, (uint) input2D.n_rows, BIT_ZERO);
    for (uint row = 0; row < this->bm_height; ++row) {
        for (uint col = 0; col < this->bm_width; ++col) {
            this->setValueAt(row, col, (uint8) input2D(row, col));
        }
    }
}

// Copy Constructor
BinaryMatrix::BinaryMatrix(const BinaryMatrix &other) {
    this->bm_width = other.bm_width;
    this->bm_height = other.bm_height;
    this->bm_baseBitSize = other.bm_baseBitSize;
    this->bm_transposed = other.bm_transposed;
    this->bm_dataLength = other.bm_dataLength;
    // Deep copy of data
    this->bm_data = new uint8[this->bm_dataLength];
    for (int i = 0; i < this->bm_dataLength; ++i) {
        this->bm_data[i] = other.bm_data[i];
    }
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

std::vector<uint> BinaryMatrix::randIndices(uint highest, uint n) {
    if (n <= 0) {
        n = rand() % highest;
    }
    if (n >= highest) {
        throw std::invalid_argument("[BinaryMatrix::randIndices] #random elements, n (arg2) should be positive and lower than (arg1)");
    }
    std::vector<uint> indices;
    indices.reserve(n);
    uint total = 0;
    while (total != n) {
        uint idx = rand() % highest;
        bool used = false;
        for (uint j : indices) {
            if (idx == j) {
                used = true;
                break;
            }
        }
        if (!used) {
            indices.emplace_back(idx);
            ++total;
        }
    }
    return indices;
}

arma::umat BinaryMatrix::randomArmaUMat(uint rows, uint cols) {
    arma::umat input2D(rows, cols);
    input2D.zeros();
    uint totalElems = rows * cols;
    std::vector<uint> indices = BinaryMatrix::randIndices(totalElems, totalElems/2);
    uint row = 0, col = 0;
    for (uint idx : indices) {
        col = idx % cols;
        row = idx / cols;
        input2D(row, col) = 1;
    }
    return input2D;
}

arma::mat BinaryMatrix::randomArmaMat(uint rows, uint cols) {
    arma::mat input2D(rows, cols);
    input2D.zeros();
    uint totalElems = rows * cols;
    std::vector<uint> indices = BinaryMatrix::randIndices(totalElems);
    uint row = 0, col = 0;
    for (uint idx : indices) {
        col = idx % cols;
        row = idx / cols;
        input2D(row, col) = arma::randn();
    }
    return input2D;
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
        return (elem & ~mask);
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
    this->bm_data[pos.first] = this->setBit(this->bm_data[pos.first], pos.second, bitValue);
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

arma::mat BinaryMatrix::bitCountPerRow(bool reshape, uint new_rows, uint new_cols) {
    arma::mat result(this->bm_height, 1);
    result.zeros();
    for (uint row = 0; row < this->bm_height; ++row) {
        for (uint col = 0; col < this->bm_width; ++col) {
            if (this->getValueAt(row, col) == BIT_ONE) {
                ++result(row, 0);
            }
        }
    }

    if (reshape) {
        // Arma is column major
        result = arma::reshape(result.t(), new_cols, new_rows).t();
    }
    return result;
}

arma::mat BinaryMatrix::bitCountPerCol(bool reshape, uint new_rows, uint new_cols) {
    arma::mat result(1, this->bm_width);
    result.zeros();
    for (uint row = 0; row < this->bm_height; ++row) {
        for (uint col = 0; col < this->bm_width; ++col) {
            if (this->getValueAt(row, col) == BIT_ONE) {
                ++result(0, col);
            }
        }
    }
    if (reshape) {
        // Arma is column major
        result = arma::reshape(result.t(), new_cols, new_rows).t();
    }
    return result;
}

std::string BinaryMatrix::uint8ToString(uint8 value) {
    std::string res = "";
    uint i = 0;
    while (i < 8) {
        res = std::to_string(value % 2) + res;
        value /= 2;
        ++i;
    }
    return res;
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
        if (this->bm_width != other.bm_height || this->bm_height != other.bm_width) {
            std::string errStr = std::string("[BinaryMatrix::operator*] Transposed input matrix dimensions ")
                  + "for this (" + std::to_string(this->bm_width) + ", " + std::to_string(this->bm_height)
                  + ") and other ("+ std::to_string(other.bm_width) + ", " + std::to_string(other.bm_height)
                  + ") should match";
            throw std::invalid_argument(errStr.c_str());
        }
        return this->tBinMultiply(other);
    }
    else {
        if (this->bm_width != other.bm_width || this->bm_height != other.bm_height) {
            std::string errStr = std::string("[BinaryMatrix::operator*] Transposed input matrix dimensions ")
                     + "for this (" + std::to_string(this->bm_width) + ", " + std::to_string(this->bm_height)
                     + ") and other ("+ std::to_string(other.bm_width) + ", " + std::to_string(other.bm_height)
                     + ") should match";
            throw std::invalid_argument(errStr.c_str());
        }
        return this->binMultiply(other);
    }
}


BinaryMatrix BinaryMatrix::im2col(uint block_width, uint block_height,
                                  uint padding, uint stride) {
    uint block_ht_half = block_height / 2;
    uint block_wd_half = block_width / 2;
    if (padding > block_ht_half || padding > block_wd_half) {
        throw std::invalid_argument("[BinaryMatrix::im2colBinary] padding (arg3), block_width(arg1) and block_height (arg2) are invalid");
    }
    uint n = block_width * block_height;
    uint rows_out = (uint) (this->bm_height - block_height + 2 * padding);
    if (rows_out % stride) {
        throw std::invalid_argument("[BinaryMatrix::im2colBinary] block_height (arg2), padding (arg3) and stride (arg4) are invalid");
    }
    rows_out = rows_out / stride + 1;
    uint cols_out = (uint) (this->bm_width - block_width + 2 * padding);
    if (cols_out % stride) {
        throw std::invalid_argument("[BinaryMatrix::im2colBinary] block_width (arg1), padding (arg3) and stride (arg4) are invalid");
    }
    cols_out = cols_out / stride + 1;
    uint all_out = rows_out * cols_out;
    BinaryMatrix result(n, all_out);

    uint res_row = 0;
    uint start_row = 0, end_row = this->bm_height;
    uint start_col = 0, end_col = this->bm_width;
    if (padding == 0) {
        start_row = padding + block_ht_half;
        end_row = this->bm_height - padding - block_ht_half;
        start_col = padding + block_wd_half;
        end_col = this->bm_height - padding - block_wd_half;
    }
    for (uint row = start_row; row < end_row; row += stride) {
        for (uint col = start_col; col < end_col; col += stride) {
            uint res_col = 0;
            int srow_start = (row - block_ht_half);
            int srow_end = (row + block_ht_half + 1);
            int scol_start = (col - block_wd_half);
            int scol_end = (col + block_wd_half + 1);
            for (int srow = srow_start; srow < srow_end; ++srow) {
                for (int scol = scol_start; scol < scol_end; ++scol) {
                    // In general
                    // result[res_row, res_col++] = input[srow, scol];
                    if (srow >= 0 && srow < this->bm_height && scol >= 0 && scol < this->bm_width) {
                        result.setValueAt(res_row, res_col, this->getValueAt(srow, scol));
                    }
                    ++res_col;
                }
            }
            ++res_row;
        }
    }

    // Check that we capture as many blocks as we had to
    assert(res_row == all_out);

    return result;
}

BinaryMatrix BinaryMatrix::repmat(uint n_rows, uint n_cols) {
    if (n_rows <= 0 || n_cols <= 0) {
        throw std::invalid_argument("[BinaryMatrix::repmat] n_rows (arg1) and n_cols (arg2) should be positive");
    }

    uint new_height = this->bm_height * n_rows;
    uint new_width = this->bm_width * n_cols;
    BinaryMatrix result(new_width, new_height);
    for (uint row = 0; row < new_height; ++row) {
        for (uint col = 0; col < new_width; ++col) {
            uint cur_row = row % this->bm_height;
            uint cur_col = col % this->bm_width;
            result.setValueAt(row, col, this->getValueAt(cur_row, cur_col));
        }
    }

    return result;
}

BinaryMatrix BinaryMatrix::reshape(uint new_rows, uint new_cols) {
    if (new_rows <= 0 || new_cols <= 0) {
        throw std::invalid_argument("[BinaryMatrix::reshape] new_rows (arg1) and new_cols (arg2) should be positive");
    }

    uint totalElems = new_rows * new_cols;
    if (totalElems != (this->bm_width * this->bm_height)) {
        throw std::invalid_argument("[BinaryMatrix::reshape] The number of elements of the matrix shouldn't change after reshape, new shape is invalid");
    }

    BinaryMatrix result(new_cols, new_rows);
    for (uint idx = 0; idx < totalElems; ++idx) {
        result.setValueAt(idx, this->getValueAt(idx));
    }

    return result;
}

arma::umat BinaryMatrix::im2colArmaMat(arma::umat input, uint block_width, uint block_height,
                                       uint padding, uint stride) {
    int block_ht_half = block_height / 2;
    int block_wd_half = block_width / 2;
    if (padding > block_ht_half || padding > block_wd_half) {
        throw std::invalid_argument("[BinaryMatrix::im2colArmaMat] Invalid padding(arg4), block width (arg2) and block height (arg3)");
    }
    uint rows_out = (uint) (input.n_rows - block_height + 2 * padding);
    if (rows_out % stride) {
        throw std::invalid_argument("[BinaryMatrix::im2colArmaMat] Invalid stride(arg5), block height(arg3) and padding(arg4) for input (arg1)");
    }
    rows_out = rows_out / stride + 1;
    uint cols_out = (uint) (input.n_cols - block_width + 2 * padding);
    if (cols_out % stride) {
        throw std::invalid_argument("[BinaryMatrix::im2colArmaMat] Invalid stride(arg5), block width(arg2) and padding(arg4) for input (arg1)");
    }
    cols_out = cols_out / stride + 1;
    uint n = block_width * block_height;
    uint all_out = rows_out * cols_out;

    arma::umat result(all_out, n);
    result.zeros();

    uint res_row = 0;
    int row_start = 0, col_start = 0;
    int row_end = (uint) input.n_rows, col_end = (uint) input.n_cols;
    if (padding == 0) {
        row_start = block_ht_half;
        row_end = (uint) input.n_rows - block_ht_half;
        col_start = block_wd_half;
        col_end = (uint) input.n_cols - block_wd_half;
    }
    for (int row = row_start; row < row_end; row += stride) {
        for (int col = col_start; col < col_end; col += stride) {
            uint res_col = 0;
            int srow_start = row - block_ht_half;
            int srow_end = row + block_ht_half;
            int scol_start = col - block_wd_half;
            int scol_end = col + block_wd_half;
            if (srow_start >= 0 && srow_start < (int) input.n_rows
                && srow_end >= 0 && srow_end < (int) input.n_rows
                && scol_start >= 0 && scol_start < (int) input.n_cols
                && scol_end >= 0 && scol_end < (int) input.n_cols ){
                result.row(res_row) = arma::vectorise(input(arma::span(srow_start, srow_end),
                                                            arma::span(scol_start, scol_end)), 1);
            } else {
                res_col = 0;
                for (int srow = srow_start; srow < (srow_end + 1); ++srow) {
                    for (int scol = scol_start; scol < (scol_end + 1); ++scol) {
                        if (srow >= 0 && srow < (int) input.n_rows && scol >= 0 && scol < (int) input.n_cols ) {
                            result(res_row, res_col) = input(srow, scol);
                        }
                        ++res_col;
                    }
                }
            }
            ++res_row;
        }
    }

    // Check that we capture as many blocks as we had to
    assert(res_row == all_out);

    return result;
}

bool BinaryMatrix::equalsArmaMat(arma::umat armaMat) {
    bool result = (armaMat.n_rows == this->bm_height) && (armaMat.n_cols == this->bm_width);
    if (!result) {
        return false;
    }

    for (uint row = 0; row < this->bm_height; ++row) {
        for (uint col = 0; col < this->bm_width; ++col) {
            result &= (this->getValueAt(row, col) == (uint8) armaMat(row, col));
            if (!result) {
                return false;
            }
        }
    }

    return result;
}
