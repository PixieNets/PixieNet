//
// Created by Esha Uboweja on 12/4/16.
//

#pragma once

#include "BinaryLayer.h"
#include <vector>
#include <armadillo>

class bd::BinaryTensor3D {
private:
    double                    bt3_alpha;
    std::vector<BinaryLayer*> bt3_tensor;

public:
    // Initialize using a fixed value
    BinaryTensor3D(uint rows, uint cols, uint channels, uint8 value);
    // Initialize to random values
    BinaryTensor3D(uint rows, uint cols, uint channels, bool randomized=false, uint n=0);
    // Initialize using Armadillo cube
    BinaryTensor3D(arma::cube tensor);
    // Initialize using copy constructor
    BinaryTensor3D(const BinaryTensor3D &tensor);
    // Destructor
    ~BinaryTensor3D();

    // im2col(3D tensor) returns a 2D matrix
    BinaryLayer im2col(uint block_width, uint block_height, uint padding, uint stride);

    // accessor functions for members
    std::vector<BinaryLayer*> tensor() {    return bt3_tensor;  }
    double                    alpha()  {    return bt3_alpha;   }
};


#endif //PROTOTYPE_BINARYTENSOR3D_H
