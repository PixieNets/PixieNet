//
// Created by Zal on 11/19/16.
//

#include "BinaryMatrix.h"

void BinaryMatrix::BinaryMatrix(int w, int h) {
    this->width = w;
    this->height = h;
    baseSize = sizeof(char);

    int n = w*h;

}

void BinaryMatrix::~BinaryMatrix() {
    delete[] data;
}

void BinaryMatrix::T() {
    this->T = !this->T;
}

BinaryMatrix BinaryMatrix::multiply(const BinaryMatrix& other) {

}

BinaryMatrix BinaryMatrix::tMultiply(const BinaryMatrix& other) {

}

BinaryMatrix BinaryMatrix::operator*(const BinaryMatrix& lhs, const BinaryMatrix& rhs ) {
    if(!this->T) {
        return tMultiply(rhs);
    }
    else {
        return multiply(rhs);
    }
}
