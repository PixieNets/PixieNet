#pragma once

class Im2Col {
public:
    static void im2col_cpu(float* data_im,
                    int channels, int height, int width,
                    int ksize, int stride, int pad, float* data_col);

};
