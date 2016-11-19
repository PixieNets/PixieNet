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
    double       doubleMultiply(const double& other);

    BinaryMatrix operator*(const BinaryMatrix& lhs, const BinaryMatrix& rhs );

};

#endif //PROTOTYPE_BINARYMATRIX_H
