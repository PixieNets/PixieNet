//
// Created by Zal on 11/21/16.
//

#include "XnorNetUtils.h"

#include <assert.h>

/**
 * Binarizes an input double-precision 2D image slice to a 2D binary matrix
 * @param data - a 2D matrix of size H x W
 * @return 2D binary matrix representation of input image slice
 */
BinaryMatrix* XnorNetUtils::centerDataMat(arma::mat data) {
    BinaryMatrix *ch = new BinaryMatrix(data.n_rows, data.n_cols);
    arma::mat centered_data = data - mean(mean(data));
    int n_elems = data.n_rows * data.n_cols;
    for (int i = 0; i < n_elems; ++i) {
        ch->setValueAt(i, (centered_data(i) >= 0.0) ? BIT_ONE:BIT_ZERO);
    }
    return ch;
}

/**
 * Binarizes an input double-precision image to a 3D binary matrix
 * @param image - a 3D matrix of size H X W X 3
 * @return 3D binary matrix representation of input image
 */
BMatArr XnorNetUtils::binarizeImage(arma::cube image) {
    assert(image.n_slices == 3); // Load only RGB images

    BMatArr binIm = new BinaryMatrix*[3];
    for (int i = 0; i < 3; ++i) {
        binIm[i] = centerDataMat(image.slice(i));
    }

    return binIm;
}