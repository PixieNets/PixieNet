//
// Created by Zal on 11/19/16.
//

#ifndef PROTOTYPE_BINARYMATRIX_H
#define PROTOTYPE_BINARYMATRIX_H

#include <iostream>
#include <string>

class BinaryMatrix {
public:
    bool    transposed;
    int     width;
    int     height;
    int     dataLength;
    char*   data;
    int     baseSize;

public:
    BinaryMatrix(int w, int h);
    ~BinaryMatrix();

    void T();
    BinaryMatrix binMultiply(const BinaryMatrix& other);
    BinaryMatrix tBinMultiply(const BinaryMatrix& other);
    double*      doubleMultiply(const double* other);
    int          bitCount();
    std::pair<int, int> elem_accessor(int i, int rows, int cols, bool transposed);
    char get_bit(char elem, int bit_id);
    char set_bit(char elem, int bit_id, char bit);
    char getValueAt(int i);

    BinaryMatrix operator*(const BinaryMatrix& other);

    void print();
    std::string toString();
    std::string dataToString();
};

#endif //PROTOTYPE_BINARYMATRIX_H
