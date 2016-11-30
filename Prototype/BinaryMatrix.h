//
// Created by Zal on 11/19/16.
//

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <armadillo>

using namespace arma;

#define uint8 uint8_t
#define IntPair std::pair<int, int>

#define BIT_ZERO ((unsigned char) 0)
#define BIT_ONE  ((unsigned char) 1)

class BinaryMatrix {
public:
    bool    transposed;
    int     width;
    int     height;
    int     dataLength;
    uint8   *data;
    int     baseBitSize;

public:
    BinaryMatrix(int w, int h);
    BinaryMatrix(int w, int h, uint8 initVal);
    ~BinaryMatrix();

    void        init(int w, int h, uint8 initVal);
    void        T();
    BinaryMatrix binMultiply(const BinaryMatrix &other);
    BinaryMatrix tBinMultiply(const BinaryMatrix &other);
    mat         doubleMultiply(const mat &other);
    int         bitCount();

    IntPair     elemAccessor(int i, int rows, int cols, bool transposed);
    uint8       getBit(uint8 elem, int bit_id);
    uint8       setBit(uint8 elem, int bit_id, uint8 bitValue);

    int         transposeIndex(int idx);
    int         transposeIndex(int idx, int width);
    int         getLinearIndex(int row, int col, int height, int width, bool transposed);
    IntPair     getDataAccessor(int row, int col);
    uint8       getValueAt(int idx);
    uint8       getValueAt(int row, int col);
    void        setValueAt(int idx, uint8 bitValue);
    void        setValueAt(int row, int col, uint8 bitValue);

    BinaryMatrix operator*(const BinaryMatrix &other);

    void        print();
    std::string toString();
    std::string dataToString();
};
