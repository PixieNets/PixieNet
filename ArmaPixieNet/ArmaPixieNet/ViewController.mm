//
//  ViewController.m
//  ArmaPixieNet
//
//  Created by Esha Uboweja on 12/7/16.
//  Copyright © 2016 Esha Uboweja. All rights reserved.
//

#import "ViewController.h"

#include "armadillo"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    arma::mat m(5, 5);
    m.randn();
    std::cout << "m = \n" << m << std::endl;
    
    arma::mat n(3, 3);
    n.randn();
    std::cout << "n = \n" << n << std::endl;
    
    arma::mat result = arma::conv2(m, n, "same");
    std::cout << "result = \n" << result << std::endl;
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
