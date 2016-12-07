//
// Created by Zal on 11/19/16.
//

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include "armadillo.h"

using namespace arma;

typedef uint8_t uint8;
typedef unsigned int uint;
typedef std::pair<uint, uint> uIntPair;

#define BIT_ZERO ((uint8) 0)
#define BIT_ONE  ((uint8) 1)

class BinaryMatrix {
public:
    bool     bm_transposed;
    uint     bm_width;
    uint     bm_height;
    uint     bm_dataLength;
    uint8    *bm_data;
    uint     bm_baseBitSize;

public:
    BinaryMatrix(uint w, uint h, uint8 initVal);
    BinaryMatrix(uint w, uint h, bool randomized=false, uint n=0);
    BinaryMatrix(arma::umat input2D);
    // Copy Constructor
    BinaryMatrix(const BinaryMatrix &other);
    // Destructor
    ~BinaryMatrix();

    void         init(uint w, uint h, uint8 initVal);
    void         T();
    BinaryMatrix binMultiply(const BinaryMatrix &other);
    BinaryMatrix tBinMultiply(const BinaryMatrix &other);
    arma::mat    doubleMultiply(const arma::mat &other);
    uint         bitCount();
    arma::mat    bitCountPerRow(bool reshape=false, uint new_rows=0, uint new_cols=0);
    arma::mat    bitCountPerCol(bool reshape=false, uint new_rows=0, uint new_cols=0);

    uIntPair     elemAccessor(uint i, uint rows, uint cols, bool transposed);
    uint8        getBit(uint8 elem, uint bit_id);
    uint8        setBit(uint8 elem, uint bit_id, uint8 bitValue);

    uint         transposeIndex(uint idx);
    uint         transposeIndex(uint idx, uint width);
    uint         getLinearIndex(uint row, uint col, uint height, uint width, bool transposed);
    uIntPair     getDataAccessor(uint row, uint col);
    uint8        getValueAt(uint idx);
    uint8        getValueAt(uint row, uint col);
    void         setValueAt(uint idx, uint8 bitValue);
    void         setValueAt(uint row, uint col, uint8 bitValue);

    bool         equalsArmaMat(arma::umat armaMat);

    // Operations on Binary Matrices
    BinaryMatrix operator*(const BinaryMatrix &other);
    BinaryMatrix im2col(uint block_width, uint block_height, uint padding, uint stride);
    BinaryMatrix repmat(uint n_rows, uint n_cols);
    BinaryMatrix reshape(uint new_rows, uint new_cols);

    void                      print();
    std::string               toString();
    std::string               dataToString();
    std::string               uint8ToString(uint8 value);

    // Static methods
    static arma::mat          randomArmaMat(uint rows, uint cols);
    static arma::umat         randomArmaUMat(uint rows, uint cols);
    static std::vector<uint>  randIndices(uint highest, uint n=0);
    static arma::umat         im2colArmaMat(arma::umat input, uint block_width, uint block_height,
                                            uint padding, uint stride);
    static arma::umat         armaXNOR(arma::umat left, arma::umat right);


    // Accessor functions for class members
    bool transposed()  {    return bm_transposed;  }
    uint width()       {    return bm_width;       }
    uint height()      {    return bm_height;      }
    uint dataLength()  {    return bm_dataLength;  }
    uint baseBitSize() {    return bm_baseBitSize; }
    uint8* data()      {    return bm_data;        }
};
