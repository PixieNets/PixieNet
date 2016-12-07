//
//  ViewController.m
//  Binary_vs_Arma
//
//  Created by Zal on 12/6/16.
//  Copyright © 2016 Salvador Medina. All rights reserved.
//

#define ARMA_DONT_USE_WRAPPER

#import "ViewController.h"

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


- (void)viewDidLoad {
    [super viewDidLoad];
    
    // ARMA MULTIPLICATON
    cout << "-- TEST ARMADILLO MUTLIPLICATON" << endl;
    arma::mat A(227,227);
    arma::mat B(227,227);
    arma::mat C(227,227);
    
    const int testIters = 5000;
    
    wall_clock timer;
    timer.tic();
    for(int i=0; i<testIters; ++i) {
        if( i % 1000 == 0){
           cout<< i << endl;
        }
        C = A*B;
    }
    double armaNSecs = timer.toc();
    cout << "[Arma mutliplication]: " << armaNSecs << " seconds."<<endl;
    cout << "[Arma mutliplication]: " << armaNSecs/testIters << " seconds/cyle"<<endl;
    
    // BINARY MATRIX MULTIPLICATION
    cout << "-- TEST BINARY MATRIX MUTLIPLICATON" << endl;
    BinaryMatrix Ba(227,227, true);
    BinaryMatrix Bb(227,227, true);
    BinaryMatrix Bc(227,227);
    
    timer.tic();
    for(int i=0; i<testIters; ++i) {
        if( i % 1000 == 0){
            cout<< i << endl;
        }
            Bc = Ba * Bb;
    }
    double binaryNSecs = timer.toc();
    cout << "[BinaryMatrix mutliplication]: " << binaryNSecs << " seconds."<<endl;
    cout << "[binaryMatrix mutliplication]: " << binaryNSecs/testIters << " seconds/cyle"<<endl;
    
    
    // BINARY LAYER MULTIPLICATION
    cout << "-- TEST BINARY LAYER MUTLIPLICATON" << endl;
    BinaryLayer BLa(227,227, 0.5, true);
    BinaryLayer BLb(227,227, 0.5, true);
    BinaryLayer BLc(227,227);
    
    timer.tic();
    for(int i=0; i<testIters; ++i) {
        if( i % 1000 == 0){
            cout<< i << endl;
        }
        BLc = BLa * BLb;
    }
    double binaryLayerNSecs = timer.toc();
    cout << "[BinaryLayer mutliplication]: " << binaryLayerNSecs << " seconds."<<endl;
    cout << "[binaryLayer mutliplication]: " << binaryLayerNSecs/testIters << " seconds/cyle"<<endl;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
