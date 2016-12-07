//
// Created by Esha Uboweja on 11/22/16.
//

#include "BinaryConvolution.h"
#include "Timer.h"

#include <assert.h>
#include <math.h>

using namespace bd;
using namespace bconv;

BinaryConvolution::BinaryConvolution(uint w, uint h, uint ch, uint k, uint stride, Convolution conv_type,
                                     Nonlinearity actv_type, Pooling pool_type, uint pool_size, uint pool_stride) {

    this->init_convolution(w, h, ch, k, stride, conv_type);
    this->init_pooling(pool_type, pool_size, pool_stride);
    this->init_nonlinearity(actv_type);
}

void BinaryConvolution::init_convolution(uint w, uint h, uint ch, uint k, uint stride, Convolution conv_type) {

    if (w <= 0 || h <= 0 || ch <= 0 || k <= 0 || stride <= 0) {
        throw std::invalid_argument("[BinaryConvolution::init_convolution] Convolution weights matrix dimensions, stride should be positive");
    }
    this->bc_width = w;
    this->bc_height = h;
    this->bc_channels = ch;
    this->bc_filters = k;
    this->bc_conv_stride = stride;

    if (conv_type == Convolution::same) {
        this->bc_padding = w / 2;
    }
    if (conv_type == Convolution::valid) {
        this->bc_padding = 0;
    }
    // The convolution filter k for computing the scales of each input sub-tensor
    this->bc_box_filter = arma::ones<arma::mat>(w, h) * (1.0 / (w * h));

    // The 4D hyper-cube weights of the convolution layer
    this->bc_conv_weights.reserve(this->bc_filters);
    for (uint f = 0; f < this->bc_filters; ++f) {
        this->bc_conv_weights.emplace_back(BinaryTensor3D(this->bc_height, this->bc_width, this->bc_channels, BIT_ZERO));
    }
}

void BinaryConvolution::init_pooling(Pooling pool_type, uint pool_size, uint pool_stride) {
    if (pool_size <= 0 || pool_stride <= 0) {
        throw std::invalid_argument("[BinaryConvolution::init_pooling] Pooling kernel dimensions, stride should be positive");
    }
    if (pool_type == Pooling::none) {
        this->bc_pool = false;
    }
    this->bc_pool_type = pool_type;
    this->bc_pool_size = pool_size;
    this->bc_pool_stride = pool_stride;
}

void BinaryConvolution::init_nonlinearity(Nonlinearity actv_type) {
    if (actv_type == Nonlinearity::none) {
        this->bc_nonlinear_actv = false;
    }
    this->bc_nonlinearity = actv_type;
}

BinaryConvolution::~BinaryConvolution() {
    // Explicit destructor only for pointer members
}

arma::mat BinaryConvolution::normalizeData2D(arma::mat data) {
    double std2 = std2Arma(data);
    arma::mat norm_data = (data - arma::mean(arma::mean(data))) / std2;
    return norm_data;
}

arma::cube BinaryConvolution::normalizeData3D(arma::cube data) {
    arma::cube norm_data = arma::zeros<arma::cube>(arma::size(data));
    for (uint ch = 0; ch < data.n_slices; ++ch) {
        norm_data.slice(ch) = normalizeData2D(data.slice(ch));
    }
    return norm_data;
}

arma::mat BinaryConvolution::input2KMat(arma::cube norm_data) {
    // A = (\sum_{i=1}^n I(:, :, i)) / n
    arma::mat A = arma::mean(norm_data, 2);
    // Convolve with box filter k of size w x h of the convolution weights
    // Discuss if Box filter using integral images is a good idea (will it be faster here
    // because the input keeps changing?)
    arma::mat K = arma::conv2(A, this->bc_box_filter, "same");
    if (this->bc_conv_stride > 1) {
        // Select the elements by stride
        uint block_ht_half = this->bc_height / 2;
        uint block_wd_half = this->bc_width / 2;
        uint n_rows = (K.n_rows - this->bc_height + 2 * this->bc_padding) / this->bc_conv_stride + (K.n_rows % 2);
        uint n_cols = (K.n_cols - this->bc_width + 2 * this->bc_padding) / this->bc_conv_stride + (K.n_cols % 2);
        uint start_row = 0, end_row = K.n_rows;
        uint start_col = 0, end_col = K.n_cols;
        if (this->bc_padding == 0) {
            start_row = this->bc_padding + block_ht_half;
            end_row = K.n_rows - this->bc_padding - block_ht_half;
            start_col = this->bc_padding + block_wd_half;
            end_col = K.n_cols - this->bc_padding - block_wd_half;
        }
        uint n_elems = n_rows * n_cols;
        arma::uvec indices(n_elems);
        uint i = 0;

        for (uint col = start_col; col < end_col; col += bc_conv_stride) {
            for (uint row = start_row; row < end_row; row += bc_conv_stride) {
                indices(i++) = arma::sub2ind(arma::size(K), row, col);
            }
        }
        K = arma::reshape(K.elem(indices), n_rows, n_cols);
    }
    return K;
}

BinaryTensor3D BinaryConvolution::binarizeInput(arma::cube norm_data) {
    return BinaryTensor3D(norm_data);
}

arma::cube BinaryConvolution::doBinaryConv(BinaryTensor3D input, arma::mat K) {

    // (sign(I) xnor_conv sign(W)) xnor_prod K,w_alpha
    arma::cube output;

    if (input.cols() < this->bc_width || input.rows() < this->bc_height) {
        // result is an empty matrix
        return output;
    }

    if (input.channels() != this->bc_channels) {
        std::cerr << "[BinaryConv::doBinConv] Input (arg1) and conv weights should have the same number of channels\n";
        return output;
    }

    // Output dimensions
    uint n_filter = this->bc_height * this->bc_width;
    uint rows_out = (input.rows() - this->bc_height + 2 * this->bc_padding) / this->bc_conv_stride + (this->bc_height % 2);
    uint cols_out = (input.cols() - this->bc_width + 2 * this->bc_padding) / this->bc_conv_stride + (this->bc_width % 2);
    output = arma::zeros(rows_out, cols_out, this->bc_filters);
    output.zeros();


    /*
    // Simple for-loop implementation
    std::vector<BinaryLayer*> inputVec = input.tensor();
    for (uint f = 0; f < this->bc_filters; ++f) {
        BinaryTensor3D cur_weights = this->bc_conv_weights[f];
        for (uint ch = 0; ch < this->bc_channels; ++ch) {
            // 1 (a). Spatial column layout of input
//            printf("[BinaryConvolution::doConv] Spatial column layout ofinput\n");
            BinaryLayer col_input = inputVec[ch]->im2col(this->bc_width, this->bc_height,
                                                     this->bc_padding, this->bc_conv_stride);
            // 1. XNOR Product of input and weights;
            // 1 (b). Spatial row layout of weight filter
//            printf("[BinaryConvolution::doConv] Spatial row layout of weight filter\n");
            BinaryLayer wt_input = cur_weights.tensor()[ch]->reshape(1, n_filter).repmat(col_input.height(), 1);
            // 1 (c). XNOR product
//            printf("[BinaryConvolution::doConv] XNOR product\n");
            BinaryLayer result = col_input * wt_input;
            // 2. Bitcount and reshape
//            printf("[BinaryConvolution::doConv] Bitcount and reshape\n");
            output.slice(f) += result.binMtx()->bitCountPerRow(true, rows_out, cols_out);
        }
        // Element-wise multiply by scalar factors of input tensor and weights alpha
//        printf("[BinaryConvolution::doConv] Element-wise multiply by scalar factors of input tensor and weights alpha\n");
//        printf("[BinaryConvolution::doConv] output.slice(%llu) dims = (%llu, %llu), K dims = (%llu, %llu), cur_weights.alpha = %f\n",
//                f, output.slice(f).n_rows, output.slice(f).n_cols, K.n_rows, K.n_cols, cur_weights.alpha());
        output.slice(f) = (output.slice(f) % K) * cur_weights.alpha();
    }
    */

    // Second for-loop implementation
    std::vector<BinaryLayer*> inputVec = input.tensor();
    for (uint ch = 0; ch < this->bc_channels; ++ch) {
        // 1 (a). Spatial column layout of input
//      printf("[BinaryConvolution::doConv] Spatial column layout ofinput\n");
        BinaryLayer col_input = inputVec[ch]->im2col(this->bc_width, this->bc_height,
                                                     this->bc_padding, this->bc_conv_stride);
        for (uint f = 0; f < this->bc_filters; ++f) {
            BinaryTensor3D cur_weights = this->bc_conv_weights[f];
            // 1. XNOR Product of input and weights;
            // 1 (b). Spatial row layout of weight filter
//            printf("[BinaryConvolution::doConv] Spatial row layout of weight filter\n");
//            BinaryLayer wt_input = cur_weights.tensor()[ch]->reshape(1, n_filter).repmat(col_input.height(), 1);
            BinaryLayer wt_input = cur_weights.tensor()[ch]->reshape(n_filter, 1).repmat(1, col_input.width());
            // 1 (c). XNOR product
//            printf("[BinaryConvolution::doConv] XNOR product\n");
            BinaryLayer result = col_input * wt_input;
            // 2. Bitcount and reshape
//            printf("[BinaryConvolution::doConv] Bitcount and reshape\n");
            // Element-wise multiply by scalar factors of input tensor and weights alpha
            output.slice(f) += result.binMtx()->bitCountPerRow(true, rows_out, cols_out);
        }
    }

    /*
    // This segment of 2D convolution is for timing purposes only
    uint f = 0;
    uint ch = 0;
    BinaryLayer col_input = inputVec[ch]->im2col(this->bc_width, this->bc_height,
                                                 this->bc_padding, this->bc_conv_stride);
    BinaryTensor3D cur_weights = this->bc_conv_weights[f];
    BinaryLayer wt_input = cur_weights.tensor()[ch]->reshape(1, n_filter).repmat(col_input.height(), 1);
    BinaryLayer result = col_input * wt_input;
    output.slice(f) += result.binMtx()->bitCountPerRow(true, rows_out, cols_out);
    */
/*
    // 3D matrix multiplication implementation
    // 1 (a). Spatial column layout of input
    BinaryLayer col_input = input.im2col(this->bc_width, this->bc_height, this->bc_padding, this->bc_conv_stride);

    // 1 (b). Spatial row layout of weight filter
    // TODO: im2col for 4D tensor of weights with reshape


    // 1 (c). XNOR product
    // TODO : xnor product of 2D matrices

    // 2. Bitcount and reshape
    // TODO : bitcount + reshape of 2D matrix into 3D cube


*/
    // Element-wise multiply by scalar factors of input tensor and weights alpha
    for (uint f = 0; f < this->bc_filters; ++f) {
//        printf("[BinaryConvolution::doConv] Element-wise multiply by scalar factors of input tensor and weights alpha\n");
//        printf("[BinaryConvolution::doConv] output.slice(%llu) dims = (%llu, %llu), K dims = (%llu, %llu), cur_weights.alpha = %f\n",
//                f, output.slice(f).n_rows, output.slice(f).n_cols, K.n_rows, K.n_cols, cur_weights.alpha());
        output.slice(f) = (output.slice(f) % K) * this->bc_conv_weights[f].alpha();
    }

    return output;
}

/*
BinaryLayer BinaryConvolution::bt4_reshape(BinaryTensor4D tensor, uint new_width, uint new_height) {

    if (tensor.empty()) {
        throw std::invalid_argument("[BinaryConvolution::bt4_reshape] BT4 tensor should be non-empty");
    }

    uint rows = tensor[0].rows();
    uint cols = tensor[0].cols();
    uint n_filter = rows * cols;
    uint channels = tensor[0].channels();
    uint filters = (uint) tensor.size();

    if ((rows * cols * channels * filters) != (new_width * new_height)) {
        throw std::invalid_argument("[BinaryConvolution::bt4_reshape] #elements shouldn't change in reshape");
    }

    BinaryLayer output(new_width, new_height);
    uint start_idx = 0, end_idx = 0;
    for (uint ch = 0; ch < channels; ++ch) {
        for (uint f = 0; f < filters; ++f) {
            BinaryTensor3D cur_weights = tensor[f];
            BinaryLayer wt_input = tensor[f].tensor()[ch]->reshape(1, n_filter);
        }
    }

    BinaryLayer result(new_width, new_height);
    return result;
}
*/


arma::cube BinaryConvolution::armaBinaryConv(arma::ucube input, arma::mat K, ArmaUTensor4D weights, uint stride,
                                             Convolution conv_type, std::vector<double> alphaPerFilter) {
    if (input.empty()) {
        throw std::invalid_argument("[BinaryConvolution::armaBinaryConv] 3D Arma Input cube should be non-empty");
    }
    if (weights.empty()) {
        throw std::invalid_argument("[BinaryConvolution::armaBinaryConv] 4D Arma Weights tensor should be non-empty");
    }
    if (weights.size() != input.n_slices) {
        throw std::invalid_argument("[BinaryConvolution::armaBinaryConv] #channels in 3D input(dim3) must equal #channels in 4D weights (dim3)");
    }

    uint rows_in = (uint) input.n_rows;
    uint cols_in = (uint) input.n_cols;
    uint channels = (uint) input.n_slices;
    uint filters = (uint) weights.size();
    uint filter_width = (uint) weights[0].n_cols;
    uint filter_height = (uint) weights[0].n_rows;
    uint padding = 0;
    if (conv_type == Convolution::same) {
        padding = filter_width / 2;
    }

    // Output dimensions
    uint n_filter = filter_width * filter_height;
    uint rows_out = (rows_in - filter_height + 2 * padding) / stride + 1;
    uint cols_out = (cols_in - filter_width + 2 * padding) / stride + 1;
    arma::cube output = arma::cube(rows_out, cols_out, filters);
    output.zeros();

    // Simple for-loop implementation
    for (uint f = 0; f < filters; ++f) {
        arma::ucube cur_weights = weights[f];
        for (uint ch = 0; ch < channels; ++ch) {
            // 1 (a). Spatial column layout of input
            arma::umat col_input = BinaryMatrix::im2colArmaMat(input.slice(ch), filter_width, filter_height,
                                                               padding, stride);
            // 1. XNOR Product of input and weights;
            // 1 (b). Spatial row layout of weight filter
            arma::umat wt_input = BinaryMatrix::im2colArmaMat(cur_weights.slice(ch), filter_width, filter_height,
                                                              padding, stride);
            // 1 (c). XNOR product
            arma::umat result = BinaryMatrix::armaXNOR(col_input, wt_input);
            // 2. Bit count and reshape
            output.slice(f) += BinaryMatrix::bitCountPerRowArma(result, true, rows_out, cols_out);
        }
    }


    for (uint f = 0; f < filters; ++f) {
        output.slice(f) = (output.slice(f) % K) * alphaPerFilter[f];
    }

    return output;
}



arma::cube BinaryConvolution::nonLinearActivate(arma::cube data) {
    arma::cube output = data;
    if (this->bc_nonlinearity == Nonlinearity::relu) {
        output.elem(arma::find(data < 0)).zeros();
    }
    return output;
}

arma::mat BinaryConvolution::poolMat(arma::mat data) {
    uint width = (uint) (data.n_cols - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint height = (uint) (data.n_rows - this->bc_pool_size) / this->bc_pool_stride - 1;

    arma::mat output = arma::zeros<arma::mat>(height, width);
    for (uint row = 0; row < height; row += this->bc_pool_stride) {
        for (uint col = 0; col < width; col += this->bc_pool_stride) {
            uint row_start = row * this->bc_pool_size;
            uint row_end = row_start + this->bc_pool_size - 1;
            uint col_start = col * this->bc_pool_size;
            uint col_end = col_start + this->bc_pool_size - 1;
            if (this->bc_pool_type == Pooling::max) {
                output(row, col) = arma::max(arma::max(data(arma::span(row_start, row_end),
                                                            arma::span(col_start, col_end))));
            } else if (this->bc_pool_type == Pooling::min) {
                output(row, col) = arma::min(arma::min(data(arma::span(row_start, row_end),
                                                            arma::span(col_start, col_end))));
            } else if (this->bc_pool_type == Pooling::average) {
                output(row, col) = arma::mean(arma::mean(data(arma::span(row_start, row_end),
                                                              arma::span(col_start, col_end))));
            }
        }
    }
    return output;
}

arma::cube BinaryConvolution::doPooling(arma::cube data) {
    uint width = (uint) (data.n_cols - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint height = (uint) (data.n_rows - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint channels = (uint) data.n_slices;

    arma::cube output = arma::zeros<arma::cube>(height, width, channels);
    for (uint ch = 0; ch < channels; ++ch) {
        output.slice(ch) = poolMat(data.slice(ch));
    }
    return output;
}

void BinaryConvolution::setWeights(BinaryTensor4D conv_weights) {
    if (conv_weights.empty()) {
        throw std::invalid_argument("[BinaryConvolution::setWeights] Input set of conv_weights must be non-empty");
    }
    this->bc_width = conv_weights[0].cols();
    this->bc_height = conv_weights[0].rows();
    this->bc_channels = conv_weights[0].channels();
    this->bc_filters = (uint) conv_weights.size();
    this->bc_conv_weights.reserve(this->bc_channels);
    for (uint f = 0; f < this->bc_filters; ++f) {
        this->bc_conv_weights.emplace_back(BinaryTensor3D(conv_weights[f]));
    }
}

arma::cube BinaryConvolution::forwardPass(arma::cube data) {

    if (data.empty()) {
        throw std::invalid_argument("[BinaryConvolution::forwardPass] Input data must be non-empty!");
    }
    if (data.n_slices != this->bc_channels) {
        std::string err = std::string("BinaryConvolution::forwardPass] Input data #channels = ")
                          + std::to_string(data.n_slices)
                          + std::string(" must match convolution weights channels = ")
                          + std::to_string(this->bc_channels);
    }

    // 1. Normalize input
    printf("[BinaryConvolution::forwardPass] Step 1. Normalize data(%llu, %llu, %llu) ...\n", data.n_rows, data.n_cols, data.n_slices);
    Utility::Timer timerStep1;
    arma::cube norm_data = this->normalizeData3D(data);
    printf("[BinaryConvolution::forwardPass] Step 1. Normalization done in time {%llu ms}\n", timerStep1.elapsedMs().count());
//    printf("[BinaryConvolution::forwardPass] Step 1. Normalization done\n");

    // 2. Generate K matrix containing scalar factors for each input sub-tensor
    printf("[BinaryConvolution::forwardPass] Step 2. Compute K matrix for data(%llu, %llu, %llu) ...\n",
           norm_data.n_rows, norm_data.n_cols, norm_data.n_slices);
    Utility::Timer timerStep2;
    arma::mat K = this->input2KMat(norm_data);
    printf("[BinaryConvolution::forwardPass] Step 2. Computed K  of size = (%llu, %llu), done in time {%llu ms}\n",
           K.n_rows, K.n_cols, timerStep2.elapsedMs().count());
//    printf("[BinaryConvolution::forwardPass] Step 2. Computed K  of size = (%llu, %llu)\n", K.n_rows, K.n_cols);

    // 3. Binarize normalized data - activation
    printf("[BinaryConvolution::forwardPass] Step 3. Activate! Converting normalized input to a 3D binary tensor...\n");
    Utility::Timer timerStep3;
    BinaryTensor3D input(norm_data);
    printf("[BinaryConvolution::forwardPass] Step 3. Activation compltete! Binary Tensor 3D of size (%d, %d, %d) in time {%llu ms}\n",
            input.rows(), input.cols(), input.channels(), timerStep3.elapsedMs().count());
//    printf("[BinaryConvolution::forwardPass] Step 3. Activation compltete! Binary Tensor 3D of size (%llu, %llu, %llu)\n",
//           input.rows(), input.cols(), input.channels());

    // 4. Perform the binary convolution
    printf("[BinaryConvolution::forwardPass] Step 4. Performing binary convolution ... \n");
    Utility::Timer timerStep4;
    arma::cube result = this->doBinaryConv(input, K);
    printf("[BinaryConvolution::forwardPass] Step 4. Binary convolution Done! Ooo lalala, output of size (%llu, %llu, %llu) in time {%llu ms}\n",
            result.n_rows, result.n_cols, result.n_slices, timerStep4.elapsedMs().count());
//    printf("[BinaryConvolution::forwardPass] Step 4. Binary convolution Done! Ooo lalala, output of size (%llu, %llu, %llu)\n",
//           result.n_rows, result.n_cols, result.n_slices);
    // 5. Apply non-linearity
    if (this->bc_nonlinear_actv) {
        printf("[BinaryConvolution::forwardPass] Step 5. Non linear activation, a moment of silence for the negative guys...\n");
        Utility::Timer timerStep5;
        result = this->nonLinearActivate(result);
        printf("[BinaryConvolution::forwardPass] Step 5. Non linear activation done! Life is full of positivity in {%llu ms}\n",
               timerStep5.elapsedMs().count());
//        printf("[BinaryConvolution::forwardPass] Step 5. Non linear activation done! Life is full of positivity\n");
    }
    // 6. Apply pooling
    if (this->bc_pool) {
        printf("[BinaryConvolution::forwardPass] Step 6. Pooling ... coz `living life, king size` isn't possible for a deep net...\n");
        Utility::Timer timerStep6;
        result = this->doPooling(result);
        printf("[BinaryConvolution::forwardPass] Step 6. Pooled result dimensions = (%llu, %llu, %llu) in {%llu ms}\n",
               result.n_rows, result.n_cols, result.n_slices, timerStep6.elapsedMs().count());
//        printf("[BinaryConvolution::forwardPass] Step 6. Pooled result dimensions = (%llu, %llu, %llu)\n",
//               result.n_rows, result.n_cols, result.n_slices);
    }

    return result;
}

double BinaryConvolution::std2Arma(arma::mat input) {
    uint n = input.n_rows * input.n_cols;
    double meanValue = arma::accu(input) / n;
    arma::mat elems = ((input - meanValue) % (input - meanValue)) / (n - 1.0);
    double result = sqrt(arma::accu(elems));
    return result;
}

ArmaUTensor4D BinaryConvolution::randomTensor4DUArma(uint width, uint height, uint channels, uint filters) {
    ArmaUTensor4D result;
    result.reserve(filters);
    for (uint f = 0; f < filters; ++f) {
        result.emplace_back(BinaryTensor3D::randomArmaUCube(height, width, channels));
    }
    return result;
}

BinaryTensor4D BinaryConvolution::uarmaToBT4(ArmaUTensor4D input) {
    BinaryTensor4D result;
    result.reserve(input.size());
    for (uint f = 0; f < input.size(); ++f) {
        result.emplace_back(BinaryTensor3D(input[f]));
    }
    return result;
}

BinaryTensor4D BinaryConvolution::randomTensor4D(uint width, uint height, uint channels, uint filters, uint nrandom) {
    BinaryTensor4D result;
    result.reserve(filters);
    for (uint f = 0; f < filters; ++f) {
        result.emplace_back(BinaryTensor3D(height, width, channels, 1.0, true, nrandom));
    }
    return result;
}

std::string BinaryConvolution::bt4ToString(BinaryTensor4D input) {
    std::string result = "";
    for (uint f = 0; f < input.size(); ++f) {
        result += "[FILTER #" + std::to_string(f) + "]\n";
        result += input[f].toString() + "\n";
    }
    return result;
}
