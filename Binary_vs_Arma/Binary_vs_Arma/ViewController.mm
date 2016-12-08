//
//  ViewController.m
//  Binary_vs_Arma
//
//  Created by Zal on 12/6/16.
//  Copyright © 2016 Salvador Medina. All rights reserved.
//

#define ARMA_DONT_USE_WRAPPER

#import "ViewController.h"

#include <cstdint>
#import "armadillo"
#import "BinaryMatrix.h"
#import "BinaryLayer.h"
#import "im2col.h"
#include "GEMM.h"
#include <stdlib.h> // Include the standard library
#include <math.h>

using namespace std;
using namespace arma;
using namespace bd;

@interface ViewController ()

@end

@implementation ViewController

typedef struct {
    //Input
    float   *input;
    int     w;
    int     h;
    int     c;
    //Output
    float   *output;
    int     out_w;
    int     out_h;
    int     out_c;
    // Update forward pass
    void    step() {
        if(input!=nullptr)  delete input;
        input = output;
        w = out_w;
        h = out_h;
        c = out_c;
    }
    int    inputSz() {
        return w*h*c;
    }
    int     outputSz() {
        return out_w*out_h*out_c;
    }
}XnorNetState;

float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

void testMultiplications() {
    // TEST SETUP
    const int mtxSz = 227;
    const int innerLoopIters = 10000;
    const int outerLoopIters = 10;
    const int totalOps = innerLoopIters * outerLoopIters;
    
    cout << "Matrix Size: " << mtxSz <<"x"<<mtxSz << endl;
    cout << "InnerLoop:   " << innerLoopIters << endl;
    cout << "OuterLoops:  " << outerLoopIters << endl;
    cout << endl;
    
    
    // ARMA MULTIPLICATON
    cout << "-- TEST ARMADILLO MUTLIPLICATON" << endl;
    arma::mat A(mtxSz,mtxSz);
    arma::mat B(mtxSz,mtxSz);
    arma::mat C(mtxSz,mtxSz);
    
    wall_clock  timer;
    double      accTime = 0;
    cout << "[Arma mutliplication]: " << endl;
    for(int l=0; l<outerLoopIters;++l) {
        timer.tic();
        for(int i=0; i<innerLoopIters; ++i) {
            C = A*B;
        }
        double nSecs = timer.toc();
        cout << nSecs/innerLoopIters << endl;
        accTime += nSecs;
    }
    cout << "[Arma Multiplication]: " << accTime / totalOps << "(secs/op)" << endl << endl;
    
    // ARMA HADDAMARD PRODUCT
    cout << "[Arma Haddamard Product]: " << endl;
    double alpha, beta, gamma;
    accTime = 0;
    for(int l=0; l<outerLoopIters;++l) {
        timer.tic();
        for(int i=0; i<innerLoopIters; ++i) {
            C = A % B;
            gamma = alpha * beta;
        }
        double nSecs = timer.toc();
        cout << nSecs/innerLoopIters << endl;
        accTime += nSecs;
    }
    cout << "[Arma Haddamard Product]: " << accTime / totalOps << "(secs/op)" << endl << endl;
    
    // BINARY MATRIX MULTIPLICATION
    cout << "-- TEST BINARY MATRIX MUTLIPLICATON" << endl;
    BinaryMatrix Ba(mtxSz,mtxSz, true);
    BinaryMatrix Bb(mtxSz,mtxSz, true);
    BinaryMatrix Bc(mtxSz,mtxSz);
    
    cout << "[BinaryMatrix multiplication]:" << endl;
    accTime = 0;
    for(int l=0; l<outerLoopIters;++l) {
        timer.tic();
        for(int i=0; i<innerLoopIters; ++i) {
            Bc = Ba * Bb;
        }
        double nSecs = timer.toc();
        cout << nSecs/innerLoopIters << endl;
        accTime += nSecs;
    }
    cout << "[BinaryMatrix Multiplication]: " << accTime / totalOps << "(secs/op)" << endl << endl;
    
    
    // BINARY LAYER MULTIPLICATION
    cout << "-- TEST BINARY LAYER MUTLIPLICATON" << endl;
    BinaryLayer BLa(mtxSz,mtxSz, 0.5, true);
    BinaryLayer BLb(mtxSz,mtxSz, 0.5, true);
    BinaryLayer BLc(mtxSz,mtxSz);
    
    cout << "[BinaryLayer multiplication]:" << endl;
    accTime = 0;
    for(int l=0; l<outerLoopIters;++l) {
        timer.tic();
        for(int i=0; i<innerLoopIters; ++i) {
            BLc = BLa * BLb;
        }
        double nSecs = timer.toc();
        cout << nSecs/innerLoopIters << endl;
        accTime += nSecs;
    }
    cout << "[BinaryLayer Multiplication]: " << accTime / totalOps << "(secs/op)" << endl << endl;
}

void testMatrixAccess() {
    wall_clock      timer;
    arma::mat       A(227,227);   A.randn();
    arma::mat       B(227,227);   B.randn();
    BinaryMatrix    Ba(227,227,true);
    bitset<51529>   BSa(51529);
    double          acc=0.0;
    double          res=0.0;
    
    cout << "-- TEST MATRIX ACCESS" << endl;
    timer.tic();
    for(int l=0; l<10000; ++l) {
        for(int row=0; row<227; ++row) {
            for(int col=0; col<227; ++col) {
                res+= ~(Ba.getValueAt(row, col)^ 1);
            }
        }
    }
    double runTime = timer.toc();
    cout << "Binary: " << runTime << endl;
    
    
    timer.tic();
    for(int l=0; l<10000; ++l) {
        for(int row=0; row<227; ++row) {
            for(int col=0; col<227; ++col) {
                res+= ~(BSa[row*227+col]^ 1);
            }
        }
    }
    runTime = timer.toc();
    cout << "Binary: " << runTime << endl;
    
    
    timer.tic();
    for(int l=0; l<10000; ++l) {
        for(int row=0; row<227; ++row) {
            for(int col=0; col<227; ++col) {
                acc+= A(row,col)*15.3;
            }
        }
    }
    runTime = timer.toc();
    cout << "Arma: " << runTime << endl;
}

void forwardPass() {
    
}

void testAllConvolutions() {
    cout << "-- TEST ALL CONVOLUTIONS" << endl;
    wall_clock timer, convTimer;
    
    XnorNetState    state;
    double avgTime = 0.0;
    
    for(int l=0; l < 10; ++l) {
        cout << "[testAllConvolutions]  Iter: " << l << endl;
        timer.tic();
        
        state.w = 227; state.h = 227; state.c = 3;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 11, 96, 0, 4);
//        cout << "Conv 1" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        
        state.w = 27; state.h = 27; state.c = 96;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 5, 256, 1, 2);
//        cout << "Conv 2" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        state.w = 13; state.h = 13; state.c = 256;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 3, 384, 1, 1);
//        cout << "Conv 3" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        state.w = 13; state.h = 13; state.c = 384;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 3, 384, 1, 1);
//        cout << "Conv 4" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        state.w = 13; state.h = 13; state.c = 384;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 3, 256, 1, 1);
//        cout << "Conv 5" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        state.w = 6; state.h = 6; state.c = 256;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 6, 4096, 1, 6);
//        cout << "FC 6" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        state.w = 1; state.h = 1; state.c = 4096;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 1, 4096, 0, 1);
//        cout << "FC 7" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        state.w = 1; state.h = 1; state.c = 4096;
        state.input = new float[state.inputSz()];
        runForwardLayer(&state, 1, 1000, 0, 1);
//        cout << "FC 8" << endl;
//        cout << state.out_w << ", " << state.out_h << ", " << state.out_c << endl;
        if(state.input!=nullptr) delete[] state.input;
        
        double runTime = timer.toc();
        cout << "[testAllConvolutions]: " << runTime << "secs" << endl;
        avgTime += runTime;
    }
    cout << "[testAllConvolutions]: Avg Time: " << avgTime / 10 << " (secs)" << endl;
    
}


void runForwardLayer(XnorNetState *s, int size, int m, int padding, int stride) {
    
    int k = size*size*s->c;
    // output definition
    s->out_h = (s->h - size + 2*padding )/stride + 1;
    s->out_w = (s->w - size + 2*padding )/stride + 1;
    s->out_c = m;
    int n = s->out_h*s->out_w;
    // required memory
    int outputSz = s->outputSz();
    float scale = sqrt(2./(size*size*s->c));
    
    // Generate random values
    // TODO: load weights through network state
    float *weights = new float[size*size*s->c*m];
    for(int i = 0; i < s->c*m*size*size; ++i) weights[i] = scale*rand_uniform(-1, 1);
    // This are computed internally
    float *binary_weights = new float[size*size*s->c*m];
    s->output = new float[outputSz];

    int workspace_size = s->out_h*s->out_w*size*size*s->c;
    float *workspace = new float[workspace_size];
    
    Im2Col::im2col_cpu(s->input, s->c, s->h, s->w, size, stride, padding, workspace);
    GEMM::gemm_cpu(0,0,m,n,k,1,weights,k,workspace,n,1,s->output,n);
    s->output += n*m;
    
    // Release the memory
    if(weights!=nullptr)        delete[] weights;
    if(binary_weights!=nullptr) delete[] binary_weights;
    if(workspace!=nullptr)      delete[] workspace;
}

void testConvForward() {
    cout << "-- TEST CONVLAYER GEMM" << endl;
    
    // Input definition
    int h = 227;
    int w = 227;
    int c = 3;
    
    int size = 11; //size filters
    int m = 96; //num filters
    int stride = 4;
    int padding = 0;
    int k = size*size*c;
    // output definition
    int out_h = (h - size + 2*padding )/stride + 1;
    int out_w = (w - size + 2*padding )/stride + 1;
    int out_c = m;
    int n = out_h*out_w;
    // required memory
    int inputSz = w * h * c;
    int outputSz = out_h * out_w * out_c;
    float scale = sqrt(2./(size*size*c));
    
    float *data = new float[inputSz];
    for(int i = 0; i < inputSz; ++i) data[i] = scale*rand_uniform(-1, 1);
    
    float *weights = new float[size*size*c*m];
    for(int i = 0; i < size*size*c*m; ++i) weights[i] = scale*rand_uniform(-1, 1);
    
    float *binary_weights = new float[size*size*c*m];
    float *bin_input = new float[inputSz];
    float *output = new float[outputSz];
    
    
    int workspace_size = out_h*out_w*size*size*c;
    float *workspace = new float[workspace_size];
    
    wall_clock timer;
    timer.tic();
    Im2Col::im2col_cpu(data, c, h, w, size, stride, padding, workspace);
    GEMM::gemm_cpu(0,0,m,n,k,1,weights,k,workspace,n,1,output,n);
    double nSecs = timer.toc();
    cout << "Convolution took: " << nSecs << endl;
    
    // Release the memory
    if(weights!=nullptr)        delete[] weights;
    if(binary_weights!=nullptr) delete[] binary_weights;
    if(bin_input!=nullptr)      delete[] bin_input;
    if(output!=nullptr)         delete[] output;
    if(workspace!=nullptr)      delete[] workspace;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    //testMultiplications();
    //testMatrixAccess();
    //testConvForward();
    testAllConvolutions();
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
