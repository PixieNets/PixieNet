//
// Created by Esha Uboweja on 11/22/16.
//

#include "BinaryConvolution.h"

#include <assert.h>

BinaryConvolution::BinaryConvolution(int w, int h, int ch, bool pool) {
    assert(w > 0 && h > 0 && ch > 0);
    this->bc_width = w;
    this->bc_height = h;
    this->bc_channels = ch;
    this->bc_pool = pool;

    this->bc_conv_weights = new BinaryLayer*[ch];
    for (int i = 0; i < ch; ++i) {
        this->bc_conv_weights[i] = new BinaryLayer(w, h);
    }
}