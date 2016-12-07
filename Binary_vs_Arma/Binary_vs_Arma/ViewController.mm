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
#include <stdlib.h> // Include the standard library

using namespace std;
using namespace arma;
using namespace bd;

@interface ViewController ()

@end

@implementation ViewController


void TestMultiplications() {
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

void TestMatrixAccess() {
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

- (void)viewDidLoad {
    [super viewDidLoad];
    //TestMultiplications();
    TestMatrixAccess();
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
