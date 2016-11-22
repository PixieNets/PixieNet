//
// Created by Zal on 11/19/16.
//

#pragma once

#include <iostream>
#include <string>
#include <armadillo>

using namespace arma;

#define uchar unsigned char
#define IntPair std::pair<int, int>

#define BIT_ZERO ((unsigned char) 0)
#define BIT_ONE  ((unsigned char) 1)

class BinaryMatrix {
public:
    bool    transposed;
    int     width;
    int     height;
    int     dataLength;
    uchar   *data;
    int     baseSize;

public:
    BinaryMatrix(int w, int h);
    BinaryMatrix(int w, int h, uchar initVal);
    ~BinaryMatrix();

    void init(int w, int h, uchar initVal);
    void T();
    BinaryMatrix binMultiply(const BinaryMatrix &other);
    BinaryMatrix tBinMultiply(const BinaryMatrix &other);
    mat          doubleMultiply(const mat &other);
    int          bitCount();

    IntPair     elemAccessor(int i, int rows, int cols, bool transposed);
    uchar       getBit(uchar elem, int bit_id);
    uchar       setBit(uchar elem, int bit_id, uchar bitValue);

    int         transposeIndex(int idx);
    int         getLinearIndex(int row, int col, int height, int width, bool transposed);
    IntPair     getDataAccessor(int row, int col);
    uchar       getValueAt(int idx);
    uchar       getValueAt(int row, int col);
    void        setValueAt(int idx, uchar bitValue);
    void        setValueAt(int row, int col, uchar bitValue);

    BinaryMatrix operator*(const BinaryMatrix &other);

    void        print();
    std::string toString();
    std::string dataToString();
};
