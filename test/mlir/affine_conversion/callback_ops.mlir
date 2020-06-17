// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// Verify that operations using callbacks are properly converted to standard call.

// -----

// Convbias Op
// CHECK-LABEL: func @simple_convbias
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<1x1x3x3xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg1 : memref<1x1x3x3xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC2:.*]] = memref_cast %arg2 : memref<1xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC3:.*]] = memref_cast %arg3 : memref<1x1x1x1xf32> to memref<*xf32>
//       CHECK: call @callback_3_inputs(%[[MC0]], %[[MC1]], %[[MC2]], %[[MC3]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_convbias(%arg0: !ng.tensor<1x1x3x3xf32>, %arg1: !ng.tensor<1x1x3x3xf32>, %arg2: !ng.tensor<1xf32>) -> !ng.tensor<1x1x1x1xf32> {
  %0 = "ng.convBias"(%arg0, %arg1, %arg2) {dilation = [1, 1], padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1], withRelu = false} : (!ng.tensor<1x1x3x3xf32>, !ng.tensor<1x1x3x3xf32>, !ng.tensor<1xf32>) -> !ng.tensor<1x1x1x1xf32>
  "ng.return"(%0) : (!ng.tensor<1x1x1x1xf32>) -> ()
}

// -----

// Softmax Op
// CHECK-LABEL: func @simple_softmax
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<2x3xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg2 : memref<2x3xf32> to memref<*xf32>
//       CHECK: call @callback_1_input(%[[MC0]], %[[MC1]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_softmax(%arg0: !ng.tensor<2x3xf32>, %arg1: !ng.tensor<1x!ng.i64>) -> !ng.tensor<2x3xf32> {
  %0 = "ng.softmax"(%arg0) {axes = [0]} : (!ng.tensor<2x3xf32>) -> !ng.tensor<2x3xf32>
  "ng.return"(%0) : (!ng.tensor<2x3xf32>) -> ()
}

// -----

// Gemm Op
// CHECK-LABEL: func @simple_gemm
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<3x6xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg1 : memref<6x4xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC2:.*]] = memref_cast %arg2 : memref<3x4xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC3:.*]] = memref_cast %arg3 : memref<3x4xf32> to memref<*xf32>
//       CHECK: call @callback_3_inputs(%[[MC0]], %[[MC1]], %[[MC2]], %[[MC3]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_gemm(%arg0: !ng.tensor<3x6xf32>, %arg1: !ng.tensor<6x4xf32>, %arg2: !ng.tensor<3x4xf32>) -> !ng.tensor<3x4xf32> {
  %0 = "ng.gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = false, transB = false} : (!ng.tensor<3x6xf32>, !ng.tensor<6x4xf32>, !ng.tensor<3x4xf32>) -> !ng.tensor<3x4xf32>
  "ng.return"(%0) : (!ng.tensor<3x4xf32>) -> ()
}

// -----

// MatMul Op
// CHECK-LABEL: func @simple_matmul
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<3x2xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg1 : memref<2x3xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC2:.*]] = memref_cast %arg2 : memref<2x2xf32> to memref<*xf32>
//       CHECK: call @callback_2_inputs(%[[MC0]], %[[MC1]], %[[MC2]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_matmul(%arg0: !ng.tensor<3x2xf32>, %arg1: !ng.tensor<2x3xf32>) -> !ng.tensor<2x2xf32> {
  %0 = "ng.matmul"(%arg0, %arg1) {transposeA = true, transposeB = true} : (!ng.tensor<3x2xf32>, !ng.tensor<2x3xf32>) -> !ng.tensor<2x2xf32>
  "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// -----

// AvePool Op
// CHECK-LABEL: func @simple_avgpool
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<2x1x3x3xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg1 : memref<2x1x3x3xf32> to memref<*xf32>
//       CHECK: call @callback_1_input(%[[MC0]], %[[MC1]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_avgpool(%arg0: !ng.tensor<2x1x3x3xf32>) -> !ng.tensor<2x1x3x3xf32> {
  %0 = "ng.avgPool"(%arg0) {includePadding = true, padAbove = [1, 1], padBelow = [0, 0], windowMovementStrides = [1, 1], windowShape = [2, 2]} : (!ng.tensor<2x1x3x3xf32>) -> !ng.tensor<2x1x3x3xf32>
  "ng.return"(%0) : (!ng.tensor<2x1x3x3xf32>) -> ()
}

// -----

// AvgPoolBackprop Op
// CHECK-LABEL: func @simple_avgpoolbackprop
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<2x2x2x2xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg1 : memref<2x2x3x3xf32> to memref<*xf32>
//       CHECK: call @callback_1_input(%[[MC0]], %[[MC1]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_avgpoolbackprop(%arg0: !ng.tensor<2x2x2x2xf32>) -> !ng.tensor<2x2x3x3xf32> {
  %0 = "ng.avgPoolBackprop"(%arg0) {forwardArgShape = [2, 2, 3, 3], includePadding = false, padAbove = [0, 0], padBelow = [0, 0], windowMovementStrides = [1, 1], windowShape = [2, 2]} : (!ng.tensor<2x2x2x2xf32>) -> !ng.tensor<2x2x3x3xf32>
  "ng.return"(%0) : (!ng.tensor<2x2x3x3xf32>) -> ()
}

// -----

// MaxPool Op
// CHECK-LABEL: func @simple_maxpool
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<64x3x7x8x10xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg1 : memref<64x3x9x6x5xf32> to memref<*xf32>
//       CHECK: call @callback_1_input(%[[MC0]], %[[MC1]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_maxpool(%arg0: !ng.tensor<64x3x7x8x10xf32>) -> !ng.tensor<64x3x9x6x5xf32> {
  %0 = "ng.maxPool"(%arg0) {padAbove = [6, 4, 5], padBelow = [5, 6, 4], windowMovementStrides = [2, 3, 4], windowShape = [2, 3, 2]} : (!ng.tensor<64x3x7x8x10xf32>) -> !ng.tensor<64x3x9x6x5xf32>
  "ng.return"(%0) : (!ng.tensor<64x3x9x6x5xf32>) -> ()
}

// -----

// MaxPoolBackprop Op
// CHECK-LABEL: func @simple_maxpoolbackprop
//       CHECK-DAG: %[[GA0:.*]] = llvm.mlir.addressof @{{[a-zA-Z_][a-zA-Z0-9_]*}} : !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">
//       CHECK-DAG: %[[C0:.*]] = constant {{[0-9]+}} : i64
//       CHECK-DAG: %[[MC0:.*]] = memref_cast %arg0 : memref<2x2x5x5xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC1:.*]] = memref_cast %arg1 : memref<2x2x4x3xf32> to memref<*xf32>
//       CHECK-DAG: %[[MC2:.*]] = memref_cast %arg2 : memref<2x2x5x5xf32> to memref<*xf32>
//       CHECK: call @callback_2_inputs(%[[MC0]], %[[MC1]], %[[MC2]], %[[GA0]], %[[C0]]) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, !llvm<"{ i8, [3 x i64], [3 x i64], [3 x i64], [3 x i64] }*">, i64) -> ()
func @simple_maxpoolbackprop(%arg0: !ng.tensor<2x2x5x5xf32>, %arg1: !ng.tensor<2x2x4x3xf32>) -> !ng.tensor<2x2x5x5xf32> {
  %0 = "ng.maxPoolBackprop"(%arg0, %arg1) {padAbove = [0, 0], padBelow = [0, 0], windowMovementStrides = [1, 1], windowShape = [2, 3]} : (!ng.tensor<2x2x5x5xf32>, !ng.tensor<2x2x4x3xf32>) -> !ng.tensor<2x2x5x5xf32>
  "ng.return"(%0) : (!ng.tensor<2x2x5x5xf32>) -> ()
}
