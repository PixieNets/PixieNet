//
// Created by Zal on 11/19/16.
//

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <armadillo>

using namespace arma;

typedef uint8_t uint8;
typedef unsigned int uint;
typedef std::pair<uint, uint> uIntPair;

#define BIT_ZERO ((unsigned char) 0)
#define BIT_ONE  ((unsigned char) 1)

class BinaryMatrix {
public:
    bool     bm_transposed;
    uint     bm_width;
    uint     bm_height;
    uint     bm_dataLength;
    uint8    *bm_data;
    uint     bm_baseBitSize;

public:
    BinaryMatrix(uint w, uint h);
    BinaryMatrix(uint w, uint h, uint8 initVal);
    ~BinaryMatrix();

    void         init(uint w, uint h, uint8 initVal);
    void         T();
    BinaryMatrix binMultiply(const BinaryMatrix &other);
    BinaryMatrix tBinMultiply(const BinaryMatrix &other);
    mat          doubleMultiply(const mat &other);
    uint         bitCount();

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

    BinaryMatrix operator*(const BinaryMatrix &other);

    void         print();
    std::string  toString();
    std::string  dataToString();

    // Accessor functions for class members
    bool tranposed()   {    return bm_transposed;  }
    uint width()       {    return bm_width;       }
    uint height()      {    return bm_height;      }
    uint dataLength()  {    return bm_dataLength;  }
    uint baseBitSize() {    return bm_baseBitSize; }
    uint8* data()      {    return bm_data;        }
};
