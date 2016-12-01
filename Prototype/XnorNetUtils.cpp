//
// Created by Zal on 11/21/16.
//

#include "XnorNetUtils.h"

#include <assert.h>

/**
 * Binarizes an input double-precision 2D data slice to a 2D binary matrix
 * @param data - a 2D matrix of size H x W
 * @return 2D binary matrix representation of input data slice
 */
BinaryMatrix* XnorNetUtils::centerDataMat(arma::mat data) {
    BinaryMatrix *ch = new BinaryMatrix(data.n_rows, data.n_cols);
    arma::mat centered_data = (data - mean(mean(data))) / stddev(stddev(data));
    int n_elems = data.n_rows * data.n_cols;
    for (int i = 0; i < n_elems; ++i) {
        ch->setValueAt(i, (centered_data(i) >= 0.0) ? BIT_ONE:BIT_ZERO);
    }
    return ch;
}

/**
 * Binarizes an input double-precision data matrix to a 3D binary matrix
 * @param data - a 3D matrix of size H X W X N
 * @return 3D binary matrix representation of input data
 */
BinaryTensor3D XnorNetUtils::normalizeData3D(arma::cube data) {
    BinaryTensor3D binMat = new BinaryLayer*[data.n_slices];
    for (int ch = 0; ch < data.n_slices; ++ch) {
        arma::mat centered_data = (data.slice(ch) - arma::mean(mean(data.slice(ch))))
                                  / arma::stddev(arma::stddev(data.slice(ch)));
        binMat[ch] = new BinaryLayer(data.n_cols, data.n_rows);
        binMat[ch]->binarizeMat(centered_data);
    }

    return binMat;
}

arma::mat relu2D(arma::mat data) {
    // max(0, data)
    arma::mat output = data;
    output.elem( arma::find(output < 0) ).zeros();
    return output;
}

arma::cube relu3D(arma::cube data) {
    // max(0, data)
    arma::cube output = data;
    output.elem( arma::find(output < 0) ).zeros();
    return output;
}

arma::vec XnorNetUtils::softmax(arma::mat W, arma::vec prevOutput) {
    assert(W.n_rows == prevOutput.n_elem);

    vec Y = arma::exp( arma::tanh(W.t() * prevOutput) );
    return Y / arma::accu(Y);
}



