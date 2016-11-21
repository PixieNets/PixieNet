//
// Created by Zal on 11/19/16.
//

#ifndef PROTOTYPE_BINARYMATRIX_H
#define PROTOTYPE_BINARYMATRIX_H

#include <iostream>
#include <string>

#define uchar unsigned char
#define IntPair std::pair<int, int>

class BinaryMatrix {
public:
    bool    transposed;
    int     width;
    int     height;
    int     dataLength;
    uchar*  data;
    int     baseSize;

public:
    BinaryMatrix(int w, int h);
    BinaryMatrix(int w, int h, int initVal);
    ~BinaryMatrix();

    void init(int w, int h, int initVal);
    void T();
    BinaryMatrix binMultiply(const BinaryMatrix& other);
    BinaryMatrix tBinMultiply(const BinaryMatrix& other);
    double*      doubleMultiply(const double* other);
    int          bitCount();

    IntPair     elem_accessor(int i, int rows, int cols, bool transposed);
    uchar       get_bit(uchar elem, int bit_id);
    uchar       set_bit(uchar elem, int bit_id, uchar bitValue);

    int         getLinearIndex(int row, int col, int height, int width, bool transposed);
    IntPair     getDataAccessor(int row, int col);
    uchar       getValueAt(int row, int col);
    void        setValueAt(int row, int col, uchar bitValue);

    BinaryMatrix operator*(const BinaryMatrix& other);

    void print();
    std::string toString();
    std::string dataToString();
};

#endif //PROTOTYPE_BINARYMATRIX_H
