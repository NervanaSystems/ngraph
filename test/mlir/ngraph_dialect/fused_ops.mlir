// RUN: ngraph-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: ngraph-opt %s | ngraph-opt | FileCheck %s

// These tests verify the parser, builder and printer of element-wise binary ops.

// CHECK-LABEL: func @squeeze
func @add_float(%arg0: !ng.tensor<2x1x2x1xf32>, %arg1: !ng.tensor<2xi64>) -> !ng.tensor<2x2xf32> {
  // CHECK: %{{[0-9]+}} = "ng.squeeze"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x1x2x1xf32>, !ng.tensor<2xi64>) -> !ng.tensor<2x2xf32>
  %0 = "ng.squeeze"(%arg0, %arg1) : (!ng.tensor<2x1x2x1xf32>, !ng.tensor<2xi64>) -> !ng.tensor<2x2xf32>
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

