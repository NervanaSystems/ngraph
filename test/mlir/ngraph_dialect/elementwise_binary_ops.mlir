// RUN: ngraph-opt %s | FileCheck %s

// These tests verify the parser, builder and printer of element-wise binary ops.

// CHECK-LABEL: func @add_float
func @add_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
  // CHECK: ng.add
  %0 = "ng.add"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

