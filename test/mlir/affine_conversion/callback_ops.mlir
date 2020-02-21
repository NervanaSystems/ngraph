// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// Verify that operations using callbacks are properly converted to standard call.

// -----

// Softmax Op
// CHECK-LABEL: func @simple_softmax
//       CHECK: %[[C1:.*]] = constant 0 : i64
//       CHECK: %[[C2:.*]] = constant {{[0-9]+}} : i64
//       CHECK: %0 = memref_cast %arg0 : memref<2x3xf32> to memref<*xf32>
//       CHECK: %1 = memref_cast %arg2 : memref<2x3xf32> to memref<*xf32>
//       CHECK: call @callback_1_input(%0, %1, %[[C1]], %[[C2]]) : (memref<*xf32>, memref<*xf32>, i64, i64) -> ()
func @simple_softmax(%arg0: !ng.tensor<2x3xf32>, %arg1: !ng.tensor<1x!ng.i64>) -> !ng.tensor<2x3xf32> {
  %0 = "ng.softmax"(%arg0) {axes = [0]} : (!ng.tensor<2x3xf32>) -> !ng.tensor<2x3xf32>
  "ng.return"(%0) : (!ng.tensor<2x3xf32>) -> ()
}

// -----

// Gemm Op
// CHECK-LABEL: func @simple_gemm
//       CHECK: %[[C1:.*]] = constant 0 : i64
//       CHECK: %[[C2:.*]] = constant {{[0-9]+}} : i64
//       CHECK: %0 = memref_cast %arg0 : memref<3x6xf32> to memref<*xf32>
//       CHECK: %1 = memref_cast %arg1 : memref<6x4xf32> to memref<*xf32>
//       CHECK: %2 = memref_cast %arg2 : memref<3x4xf32> to memref<*xf32>
//       CHECK: %3 = memref_cast %arg3 : memref<3x4xf32> to memref<*xf32>
//       CHECK: call @callback_3_inputs(%0, %1, %2, %3, %[[C1]], %[[C2]]) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, i64, i64) -> ()
func @simple_gemm(%arg0: !ng.tensor<3x6xf32>, %arg1: !ng.tensor<6x4xf32>, %arg2: !ng.tensor<3x4xf32>) -> !ng.tensor<3x4xf32> {
  %0 = "ng.gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = false, transB = false} : (!ng.tensor<3x6xf32>, !ng.tensor<6x4xf32>, !ng.tensor<3x4xf32>) -> !ng.tensor<3x4xf32>
  "ng.return"(%0) : (!ng.tensor<3x4xf32>) -> ()
}

// -----

// MatMul Op
// CHECK-LABEL: func @simple_matmul
//       CHECK: %[[C1:.*]] = constant 0 : i64
//       CHECK: %[[C2:.*]] = constant {{[0-9]+}} : i64
//       CHECK: %0 = memref_cast %arg0 : memref<3x2xf32> to memref<*xf32>
//       CHECK: %1 = memref_cast %arg1 : memref<2x3xf32> to memref<*xf32>
//       CHECK: %2 = memref_cast %arg2 : memref<2x2xf32> to memref<*xf32>
//       CHECK: call @callback_2_inputs(%0, %1, %2, %[[C1]], %[[C2]]) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i64, i64) -> ()
func @simple_matmul(%arg0: !ng.tensor<3x2xf32>, %arg1: !ng.tensor<2x3xf32>) -> !ng.tensor<2x2xf32> {
  %0 = "ng.matmul"(%arg0, %arg1) {transposeA = true, transposeB = true} : (!ng.tensor<3x2xf32>, !ng.tensor<2x3xf32>) -> !ng.tensor<2x2xf32>
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// -----

// AvePool Op
// CHECK-LABEL: func @simple_avgpool
//       CHECK: %0 = memref_cast %arg0 : memref<2x1x3x3xf32> to memref<*xf32>
//       CHECK: %1 = memref_cast %arg1 : memref<2x1x3x3xf32> to memref<*xf32>
//       CHECK: %[[C1:.*]] = constant 0 : i64
//       CHECK: %[[C2:.*]] = constant {{[0-9]+}} : i64
//       CHECK: call @callback_1_input(%0, %1, %[[C1]], %[[C2]]) : (memref<*xf32>, memref<*xf32>, i64, i64) -> ()
func @simple_avgpool(%arg0: !ng.tensor<2x1x3x3xf32>) -> !ng.tensor<2x1x3x3xf32> {
  %0 = "ng.avgPool"(%arg0) {includePadding = true, padAbove = [1, 1], padBelow = [0, 0], windowMovementStrides = [1, 1], windowShape = [2, 2]} : (!ng.tensor<2x1x3x3xf32>) -> !ng.tensor<2x1x3x3xf32>
  "ng.return"(%0) : (!ng.tensor<2x1x3x3xf32>) -> ()
}

// -----

// AvgPoolBackprop Op
// CHECK-LABEL: func @simple_avgpoolbackprop
//       CHECK: %0 = memref_cast %arg0 : memref<2x2x2x2xf32> to memref<*xf32>
//       CHECK: %1 = memref_cast %arg1 : memref<2x2x3x3xf32> to memref<*xf32>
//       CHECK: %[[C1:.*]] = constant 0 : i64
//       CHECK: %[[C2:.*]] = constant {{[0-9]+}} : i64
//       CHECK: call @callback_1_input(%0, %1, %[[C1]], %[[C2]]) : (memref<*xf32>, memref<*xf32>, i64, i64) -> ()
func @simple_avgpoolbackprop(%arg0: !ng.tensor<2x2x2x2xf32>) -> !ng.tensor<2x2x3x3xf32> {
  %0 = "ng.avgPoolBackprop"(%arg0) {forwardArgShape = [2, 2, 3, 3], includePadding = false, padAbove = [0, 0], padBelow = [0, 0], windowMovementStrides = [1, 1], windowShape = [2, 2]} : (!ng.tensor<2x2x2x2xf32>) -> !ng.tensor<2x2x3x3xf32>
  "ng.return"(%0) : (!ng.tensor<2x2x3x3xf32>) -> ()
}

// -----

// MaxPool Op
// CHECK-LABEL: func @simple_maxpool
//       CHECK: %0 = memref_cast %arg0 : memref<64x3x7x8x10xf32> to memref<*xf32>
//       CHECK: %1 = memref_cast %arg1 : memref<64x3x9x6x5xf32> to memref<*xf32>
//       CHECK: %[[C1:.*]] = constant 0 : i64
//       CHECK: %[[C2:.*]] = constant {{[0-9]+}} : i64
//       CHECK: call @callback_1_input(%0, %1, %[[C1]], %[[C2]]) : (memref<*xf32>, memref<*xf32>, i64, i64) -> ()
func @simple_maxpool(%arg0: !ng.tensor<64x3x7x8x10xf32>) -> !ng.tensor<64x3x9x6x5xf32> {
  %0 = "ng.maxPool"(%arg0) {padAbove = [6, 4, 5], padBelow = [5, 6, 4], windowMovementStrides = [2, 3, 4], windowShape = [2, 3, 2]} : (!ng.tensor<64x3x7x8x10xf32>) -> !ng.tensor<64x3x9x6x5xf32>
  "ng.return"(%0) : (!ng.tensor<64x3x9x6x5xf32>) -> ()
}

// -----

// MaxPoolBackprop Op
// CHECK-LABEL: func @simple_maxpoolbackprop
//       CHECK: %0 = memref_cast %arg0 : memref<2x2x5x5xf32> to memref<*xf32>
//       CHECK: %1 = memref_cast %arg1 : memref<2x2x4x3xf32> to memref<*xf32>
//       CHECK: %2 = memref_cast %arg2 : memref<2x2x5x5xf32> to memref<*xf32>
//       CHECK: %[[C1:.*]] = constant 0 : i64
//       CHECK: %[[C2:.*]] = constant {{[0-9]+}} : i64
//       CHECK: call @callback_2_inputs(%0, %1, %2, %[[C1]], %[[C2]]) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, i64, i64) -> ()
func @simple_maxpoolbackprop(%arg0: !ng.tensor<2x2x5x5xf32>, %arg1: !ng.tensor<2x2x4x3xf32>) -> !ng.tensor<2x2x5x5xf32> {
  %0 = "ng.maxPoolBackprop"(%arg0, %arg1) {padAbove = [0, 0], padBelow = [0, 0], windowMovementStrides = [1, 1], windowShape = [2, 3]} : (!ng.tensor<2x2x5x5xf32>, !ng.tensor<2x2x4x3xf32>) -> !ng.tensor<2x2x5x5xf32>
  "ng.return"(%0) : (!ng.tensor<2x2x5x5xf32>) -> ()
}
