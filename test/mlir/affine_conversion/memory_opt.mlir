// RUN: ngraph-opt %s --split-input-file --ngraph-memory-opt --ngraph-memory-opt-concat --ngraph-memory-opt-eltwise  -convert-ngraph-to-affine  | FileCheck %s

//func @test0(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
//  %0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %1 = "ng.add"(%0, %0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %2 = "ng.add"(%1, %1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %3 = "ng.add"(%2, %2) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  "ng.return"(%3) : (!ng.tensor<2x2xf32>) -> ()
//}
//
//// -----
//
//func @test1(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<4x2xf32> {
//  %0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %1 = "ng.add"(%0, %0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %2 = "ng.concat"(%0, %1) {concatenation_axis = 0} : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<4x2xf32>
//  %3 = "ng.add"(%2, %2) : (!ng.tensor<4x2xf32>, !ng.tensor<4x2xf32>) -> !ng.tensor<4x2xf32>
//  "ng.return"(%3) : (!ng.tensor<4x2xf32>) -> ()
//}
//
//// -----

func @test2(%arg0: !ng.tensor<1x2x2xf32>, %arg1: !ng.tensor<1x2x2xf32>) -> (!ng.tensor<1x4x2xf32>, !ng.tensor<1x8x2xf32>){
  %0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x2x2xf32>
  %1 = "ng.add"(%arg0, %arg1) : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x2x2xf32>
  // inplace
  %2 = "ng.concat"(%0, %1) {concatenation_axis = 1} : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x4x2xf32>
  // cannot be done inplace, %3 and %2 cannot alias
  %3 = "ng.concat"(%0, %1, %2) {concatenation_axis = 1} : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>, !ng.tensor<1x4x2xf32>) -> !ng.tensor<1x8x2xf32>
  // inplace destructive. %3 and %2 cannot alias
  %4 = "ng.add"(%3, %3) : (!ng.tensor<1x8x2xf32>, !ng.tensor<1x8x2xf32>) -> !ng.tensor<1x8x2xf32>
  
  // no inplace, result is output
  %5 = "ng.add"(%2, %2) : (!ng.tensor<1x4x2xf32>, !ng.tensor<1x4x2xf32>) -> !ng.tensor<1x4x2xf32>
  // no inplace, result is output
  %6 = "ng.add"(%4, %4) : (!ng.tensor<1x8x2xf32>, !ng.tensor<1x8x2xf32>) -> !ng.tensor<1x8x2xf32>
  "ng.return"(%5, %6) : (!ng.tensor<1x4x2xf32>, !ng.tensor<1x8x2xf32>) -> ()
}

// -----
// cascaded concats test. All concats will get same buffer. 


func @test3(%arg0: !ng.tensor<1x2x2xf32>, %arg1: !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x16x2xf32> {
  %0 = "ng.concat"(%arg0, %arg1) {concatenation_axis = 1} : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x4x2xf32>
  %1 = "ng.concat"(%arg0, %arg1) {concatenation_axis = 1} : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x4x2xf32>
  %2 = "ng.concat"(%arg0, %arg1) {concatenation_axis = 1} : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x4x2xf32>
  %3 = "ng.concat"(%arg0, %arg1) {concatenation_axis = 1} : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x4x2xf32>
  %4 = "ng.concat"(%0, %1) {concatenation_axis = 1} : (!ng.tensor<1x4x2xf32>, !ng.tensor<1x4x2xf32>) -> !ng.tensor<1x8x2xf32>
  %5 = "ng.concat"(%2, %3) {concatenation_axis = 1} : (!ng.tensor<1x4x2xf32>, !ng.tensor<1x4x2xf32>) -> !ng.tensor<1x8x2xf32>
  %6 = "ng.concat"(%4, %5) {concatenation_axis = 1} : (!ng.tensor<1x8x2xf32>, !ng.tensor<1x8x2xf32>) -> !ng.tensor<1x16x2xf32>
  %7 = "ng.add"(%6, %6) : (!ng.tensor<1x16x2xf32>, !ng.tensor<1x16x2xf32>) -> !ng.tensor<1x16x2xf32>
  "ng.return"(%7) : (!ng.tensor<1x16x2xf32>) -> ()
}

// -----

func @test4(%arg0: !ng.tensor<1x2x2xf32>, %arg1: !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x8x2xf32> {
    %S0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x2x2xf32>
    %S1 = "ng.add"(%arg0, %arg1) : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x2x2xf32>
    %S2 = "ng.add"(%arg0, %arg1) : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x2x2xf32>
    %R0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x2x2xf32>
    %R2 = "ng.add"(%arg0, %arg1) : (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x2x2xf32>
    // pre-existing assignment of S1 in %D2 prevents assignment for %D1 concat
    %D1 = "ng.concat"(%S0, %S1, %S2) {concatenation_axis = 1} :  (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x6x2xf32>
    %D2 = "ng.concat"(%R0, %S1, %R2) {concatenation_axis = 1} :  (!ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>, !ng.tensor<1x2x2xf32>) -> !ng.tensor<1x6x2xf32>
    %D3 = "ng.add"(%D1, %D2)    : (!ng.tensor<1x6x2xf32>, !ng.tensor<1x6x2xf32>) -> !ng.tensor<1x6x2xf32>
    "ng.return"(%D3) : (!ng.tensor<1x6x2xf32>) -> ()
}
