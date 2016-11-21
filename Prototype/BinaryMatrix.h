//
// Created by Zal on 11/19/16.
//

#ifndef PROTOTYPE_BINARYMATRIX_H
#define PROTOTYPE_BINARYMATRIX_H

#include <iostream>
#include <string>

#define uchar unsigned char

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
    std::pair<int, int> elem_accessor(int i, int rows, int cols, bool transposed);
    uchar get_bit(uchar elem, int bit_id);
    uchar set_bit(uchar elem, int bit_id, uchar bitValue);
    uchar getValueAt(int i);
    void setValueAt(int row, int col, uchar bitValue);

    BinaryMatrix operator*(const BinaryMatrix& other);

    void print();
    std::string toString();
    std::string dataToString();
};

#endif //PROTOTYPE_BINARYMATRIX_H
