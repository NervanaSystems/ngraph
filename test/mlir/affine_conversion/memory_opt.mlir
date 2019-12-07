// RUN: ngraph-opt %s --split-input-file --ngraph-memory-opt --ngraph-memory-opt-concat --ngraph-memory-opt-eltwise  -convert-ngraph-to-affine  | FileCheck %s

// CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 * 2 + d1)
// CHECK-LABEL: test0	
// CHECK: %[[B:.*]] = alloc() : memref<16xi8>
// CHECK: std.view %[[B]][][] : memref<16xi8> to memref<2x2xf32, #[[MAP0]]>
// CHECK: std.view %[[B]][][] : memref<16xi8> to memref<2x2xf32, #[[MAP0]]>
// CHECK: std.view %[[B]][][] : memref<16xi8> to memref<2x2xf32, #[[MAP0]]>
// CHECK: dealloc %[[B]] : memref<16xi8>
func @test0(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
  %0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %1 = "ng.add"(%0, %0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %2 = "ng.add"(%1, %1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %3 = "ng.add"(%2, %2) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  "ng.return"(%3) : (!ng.tensor<2x2xf32>) -> ()
}

// -----

// CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 * 2 + d1)
// CHECK-DAG: #[[MAP1:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 * 2 + d1 + 4)
// CHECK-LABEL: test1
// CHECK: %[[B:.*]] = alloc() : memref<32xi8>
// CHECK: std.view %[[B]][][] : memref<32xi8> to memref<2x2xf32, #[[MAP0]]>
// CHECK: std.view %[[B]][][] : memref<32xi8> to memref<2x2xf32, #[[MAP1]]>
// CHECK: std.view %[[B]][][] : memref<32xi8> to memref<4x2xf32, #[[MAP0]]>
// CHECK: dealloc %[[B]] : memref<32xi8>
func @test1(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<4x2xf32> {
  %0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %1 = "ng.add"(%0, %0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %2 = "ng.concat"(%0, %1) {concatenation_axis = 0} : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<4x2xf32>
  %3 = "ng.add"(%2, %2) : (!ng.tensor<4x2xf32>, !ng.tensor<4x2xf32>) -> !ng.tensor<4x2xf32>
  "ng.return"(%3) : (!ng.tensor<4x2xf32>) -> ()
}

// -----

// CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 4 + d1 * 2 + d2)
// CHECK-DAG: #[[MAP1:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 4 + d1 * 2 + d2 + 4)
// CHECK-DAG: #[[MAP2:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 8 + d1 * 2 + d2)
// CHECK-DAG: #[[MAP3:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 16 + d1 * 2 + d2)
// CHECK-LABEL: test2
// CHECK: %[[B1:.*]] = alloc() : memref<32xi8>
// CHECK: std.view %[[B1]][][] : memref<32xi8> to memref<1x2x2xf32, #[[MAP0]]>
// CHECK: std.view %[[B1]][][] : memref<32xi8> to memref<1x2x2xf32, #[[MAP1]]>
// CHECK: std.view %[[B1]][][] : memref<32xi8> to memref<1x4x2xf32, #[[MAP2]]>
// CHECK: %[[B2:.*]] = alloc() : memref<64xi8>
// CHECK: std.view %[[B2]][][] : memref<64xi8> to memref<1x8x2xf32, #[[MAP3]]>
// CHECK: std.view %[[B2]][][] : memref<64xi8> to memref<1x8x2xf32, #[[MAP3]]>
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

// CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 8 + d1 * 2 + d2)
// CHECK-DAG: #[[MAP8:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 8 + d1 * 2 + d2 + 8)
// CHECK-DAG: #[[MAP9:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 8 + d1 * 2 + d2 + 16)
// CHECK-DAG: #[[MAP10:[a-zA-Z0-9]+]]  = (d0, d1, d2) -> (d0 * 8 + d1 * 2 + d2 + 24)
// CHECK-DAG: #[[MAP11:[a-zA-Z0-9]+]]  = (d0, d1, d2) -> (d0 * 16 + d1 * 2 + d2)
// CHECK-DAG: #[[MAP12:[a-zA-Z0-9]+]]  = (d0, d1, d2) -> (d0 * 16 + d1 * 2 + d2 + 16)
// CHECK-DAG: #[[MAP13:[a-zA-Z0-9]+]]  = (d0, d1, d2) -> (d0 * 32 + d1 * 2 + d2)
// CHECK-LABEL: test3
// CHECK: %[[B:.*]] = alloc() : memref<128xi8>
// CHECK: std.view %[[B]][][] : memref<128xi8> to memref<1x4x2xf32, #[[MAP0]]>
// CHECK: std.view %[[B]][][] : memref<128xi8> to memref<1x4x2xf32, #[[MAP8]]>
// CHECK: std.view %[[B]][][] : memref<128xi8> to memref<1x4x2xf32, #[[MAP9]]>
// CHECK: std.view %[[B]][][] : memref<128xi8> to memref<1x4x2xf32, #[[MAP10]]>
// CHECK: std.view %[[B]][][] : memref<128xi8> to memref<1x8x2xf32, #[[MAP11]]>
// CHECK: std.view %[[B]][][] : memref<128xi8> to memref<1x8x2xf32, #[[MAP12]]>
// CHECK: std.view %[[B]][][] : memref<128xi8> to memref<1x16x2xf32, #[[MAP13]]>
// CHECK: dealloc %[[B]] : memref<128xi8>
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

//CHECK-DAG: #[[MAP4:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 4 + d1 * 2 + d2 + 4)
//CHECK-DAG: #[[MAP5:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 4 + d1 * 2 + d2)
//CHECK-DAG: #[[MAP6:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 4 + d1 * 2 + d2 + 8)
//CHECK-DAG: #[[MAP12:[a-zA-Z0-9]+]] = (d0, d1, d2) -> (d0 * 12 + d1 * 2 + d2)
// CHECK-LABEL: test4
//CHECK: %[[B1:.*]] = alloc() : memref<1x2x2xf32>
//CHECK: %[[B2:.*]] = alloc() : memref<48xi8>
//CHECK: std.view %[[B2]][][] : memref<48xi8> to memref<1x2x2xf32, #[[MAP4]]>
//CHECK: %[[B3:.*]] = alloc() : memref<1x2x2xf32>
//CHECK: std.view %[[B2]][][] : memref<48xi8> to memref<1x2x2xf32, #[[MAP5]]>
//CHECK: std.view %[[B2]][][] : memref<48xi8> to memref<1x2x2xf32, #[[MAP6]]>
//CHECK: %[[B4:.*]] = alloc() : memref<1x6x2xf32>
//CHECK: std.view %1[][] : memref<48xi8> to memref<1x6x2xf32, #[[MAP12]]>
//CHECK: dealloc %[[B1]] : memref<1x2x2xf32>
//CHECK: dealloc %[[B2]] : memref<48xi8>
//CHECK: dealloc %[[B3]] : memref<1x2x2xf32>
//CHECK: dealloc %[[B4]] : memref<1x6x2xf32>
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
