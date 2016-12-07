//
// Created by Zal on 12/7/16.
//

#ifndef PROTOTYPE_BITSETCONV_H
#define PROTOTYPE_BITSETCONV_H

#include <bitset>

typedef struct {
    int width;
    int height;
    std::bitset<10> data;
} BitMatrix;

class BitSetConv {
    bool im2colGetBit(std::bitset *data_in, int height, int width, int channels,
                           int row, int col, int channel, int pad)
    {
        row -= pad;
        col -= pad;
        if (row < 0 || col < 0 ||
            row >= height || col >= width) return 0;

        return data_in[col + width*(row + height*channel)];
    }

    void im2col(std::bitset *data_in,
                int height, int width, int channels,
                int kSize, int stride, int pad, std::bitset *data_out) {
        int c,h,w;
        int height_col = (height + 2*pad - kSize) / stride + 1;
        int width_col = (width + 2*pad - kSize) / stride + 1;
        int channels_col = channels * kSize * kSize;

        for(c = 0; c < channels_col; ++c) {
            int w_offset = c % kSize;
            int h_offset = (c / kSize) % kSize;
            int c_im = c / kSize / kSize;
            for(h = 0; h < height_col; ++h) {
                for(w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col * w;

                }
            }
        }
    }
};


#endif //PROTOTYPE_BITSETCONV_H
