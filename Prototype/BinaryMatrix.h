//
// Created by Zal on 11/19/16.
//

#ifndef PROTOTYPE_BINARYTENSOR_H
#define PROTOTYPE_BINARYTENSOR_H


class BinaryMatrix {
    bool    transposed;
    int     width;
    int     height;
    char*   data;
    int     baseSize;

public:
    void BinaryMatrix(int w, int h);
    void ~BinaryMatrix();
    void T();
    BinaryMatrix multiply(const BinaryMatrix& other);
    BinaryMatrix tMultiply(const BinaryMatrix& other);


    BinaryMatrix operator*(const BinaryMatrix& lhs, const BinaryMatrix& rhs );

};


#endif //PROTOTYPE_BINARYTENSOR_H
