#include <iostream>
#include <assert.h>

#include "TestArmadillo.h"
#include "TestBinaryMatrix.h"
#include "TestBinaryLayer.h"
#include "TestBinaryTensor3D.h"
#include "TestBinaryConvolution.h"
#include "TestXnorNetwork.h"


int main() {
    //TestArmadillo testArma;
    TestBinaryMatrix testBinaryMatrix;
    TestBinaryLayer testBinaryLayer;
    TestBinaryTensor3D testBinaryTensor3D;
    TestBinaryConvolution testBinaryConvolution;
    TestXnorNetwork testXnorNetwork;

    //testArma.runTest();
    //testArma.runTest();
    testBinaryMatrix.runAllTests();
    testBinaryLayer.runAllTests();
    testBinaryTensor3D.runAllTests();
    testBinaryConvolution.runAllTests();
    testXnorNetwork.runAllTests();

    return 0;
}