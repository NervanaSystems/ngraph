// RUN: ngraph-opt %s -fuse-ngraph-dialect -split-input-file | FileCheck %s

// Verify that operations fused using pattern matcher are properly replaced with correct Fused Op.

// -----

// matmul+bias
// CHECK-LABEL:   func @matmul_bias_fusion(%arg0: !ng.tensor<2x4xf32>, %arg1: !ng.tensor<4x1xf32>, %arg2: !ng.tensor<2x1xf32>) -> !ng.tensor<2x1xf32> {
//       CHECK:   %0 = "ng.gemm"(%arg0, %arg1, %arg2) {alpha = {{.*}}: f32, beta = {{.*}} : f32, transA = {{.*}}, transB = {{.*}}} : (!ng.tensor<2x4xf32>, !ng.tensor<4x1xf32>, !ng.tensor<2x1xf32>) -> !ng.tensor<2x1xf32>
//       CHECK:   "ng.return"(%0) : (!ng.tensor<2x1xf32>) -> ()
func @matmul_bias_fusion(%arg0: !ng.tensor<2x4xf32>, %arg1: !ng.tensor<4x1xf32>, %arg2: !ng.tensor<2x1xf32>) -> !ng.tensor<2x1xf32> {
    %0 = "ng.dot"(%arg0, %arg1) : (!ng.tensor<2x4xf32>, !ng.tensor<4x1xf32>) -> !ng.tensor<2x1xf32>
    %1 = "ng.add"(%0, %arg2) : (!ng.tensor<2x1xf32>, !ng.tensor<2x1xf32>) -> !ng.tensor<2x1xf32>
    "ng.return"(%1) : (!ng.tensor<2x1xf32>) -> ()
}

