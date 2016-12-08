//
//  ViewController.m
//  ArmaPixieNet
//
//  Created by Esha Uboweja on 12/7/16.
//  Copyright © 2016 Esha Uboweja. All rights reserved.
//

#import "ViewController.h"

#include "armadillo"
#include "stdlib.h"


@interface ViewController ()

@end

@implementation ViewController

void changeMatrix(arma::mat *input, arma::uword row, arma::uword col) {
//    if (row < input->n_rows && col < input->n_cols) {
//        input->at(row, col) = 7.0;
//    }
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    arma::mat m(5, 5);
    m.randn();
    std::cout << "m = \n" << m << std::endl;
    std::cout << "changing m ...\n";
    changeMatrix(&m, 3, 3);
    std::cout << "m = \n" << m << std::endl;
    
    arma::mat n(3, 3);
    n.randn();
    std::cout << "n = \n" << n << std::endl;
    
    arma::mat result = arma::conv2(m, n, "same");
    std::cout << "result = \n" << result << std::endl;
    
    arma::cube c(3, 3, 2);
    c.randn();
    std::cout << "c = \n" << c << std::endl;
    
    arma::Cube<float> cf(3, 3, 3);
    cf.randn();
    std::cout << "cf = \n" << cf << std::endl;
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
