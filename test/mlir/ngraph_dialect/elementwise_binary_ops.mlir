// RUN: ngraph-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: ngraph-opt %s | ngraph-opt | FileCheck %s

// These tests verify the parser, builder and printer of element-wise binary ops.

// CHECK-LABEL: func @add_float
func @add_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
  // CHECK: %{{.*}} = "ng.add"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %0 = "ng.add"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// CHECK-LABEL: func @equal_float
func @equal_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8> {
  // CHECK: %{{.*}} = "ng.equal"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  %0 = "ng.equal"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// CHECK-LABEL: func @notequal_float
func @notequal_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8> {
  // CHECK: %{{.*}} = "ng.not.equal"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  %0 = "ng.not.equal"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// CHECK-LABEL: func @greater_float
func @greater_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8> {
  // CHECK: %{{.*}} = "ng.greater"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  %0 = "ng.greater"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// CHECK-LABEL: func @greatereq_float
func @greatereq_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8> {
  // CHECK: %{{.*}} = "ng.greater.eq"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  %0 = "ng.greater.eq"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// CHECK-LABEL: func @less_float
func @less_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8> {
  // CHECK: %{{.*}} = "ng.less"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  %0 = "ng.less"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// CHECK-LABEL: func @lesseq_float
func @lesseq_float(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8> {
  // CHECK: %{{.*}} = "ng.less.eq"(%{{.*}}, %{{.*}}) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  %0 = "ng.less.eq"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
  "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}
