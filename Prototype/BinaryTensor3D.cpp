//
// Created by Esha Uboweja on 12/4/16.
//

#include "BinaryTensor3D.h"

using namespace bd;

BinaryTensor3D::BinaryTensor3D(uint rows, uint cols, uint channels, uint8 value) {

}

BinaryTensor3D::BinaryTensor3D(uint rows, uint cols, uint channels, bool randomized, uint n) {

}

BinaryTensor3D::BinaryTensor3D(arma::cube tensor) {

}

BinaryTensor3D::BinaryTensor3D(const BinaryTensor3D &tensor) {

}

BinaryTensor3D::~BinaryTensor3D() {

}

BinaryLayer BinaryTensor3D::im2col(uint block_width, uint block_height, uint padding, uint stride) {
    BinaryLayer result(0,0);
    return result;
}
