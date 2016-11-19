//
// Created by Zal on 11/19/16.
//

#ifndef PROTOTYPE_BINARYMATRIX_H
#define PROTOTYPE_BINARYMATRIX_H


class BinaryMatrix {
    bool    transposed;
    int     width;
    int     height;
    int     dataLength;
    char*   data;
    int     baseSize;

public:
    void BinaryMatrix(int w, int h);
    void ~BinaryMatrix();
    void T();
    BinaryMatrix binMultiply(const BinaryMatrix& other);
    BinaryMatrix tBinMultiply(const BinaryMatrix& other);
    double*      doubleMultiply(const double* other);
    std::pair<int, int> elem_accessor(int i, int rows, int cols, bool transposed);
    char get_bit(char elem, int bit_id);
    char set_bit(char elem, int bit_id, char bit);

    BinaryMatrix operator*(const BinaryMatrix& other);

};

#endif //PROTOTYPE_BINARYMATRIX_H
