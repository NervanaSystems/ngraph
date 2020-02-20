// RUN: ngraph-opt %s -ngraph-op-fusion -split-input-file | FileCheck %s


func @matmul_bias(%arg0: !ng.tensor<512x256xf32>, %arg1: !ng.tensor<256x256xf32>) -> !ng.tensor<512x256xf32> {
  %0 = "ng.dot"(%arg0, %arg1) : (!ng.tensor<512x256xf32>, !ng.tensor<256x256xf32>) -> !ng.tensor<512x256xf32>
  %1 = "ng.add"(%0, %arg0) : (!ng.tensor<512x256xf32>, !ng.tensor<512x256xf32>) -> !ng.tensor<512x256xf32>
  "ng.return"(%1) : (!ng.tensor<512x256xf32>) -> ()
}
// CHECK-LABEL: func @matmul_bias

// -----

func @simple_elementwise(%arg0: !ng.tensor<512xf32>, %arg1: !ng.tensor<512xf32>) -> !ng.tensor<512xf32> {
  %0 = "ng.mul"(%arg0, %arg1) : (!ng.tensor<512xf32>, !ng.tensor<512xf32>) -> !ng.tensor<512xf32>
  %1 = "ng.add"(%0, %arg1) : (!ng.tensor<512xf32>, !ng.tensor<512xf32>) -> !ng.tensor<512xf32>
  %2 = "ng.relu"(%1) : (!ng.tensor<512xf32>) -> !ng.tensor<512xf32>
  "ng.return"(%2) : (!ng.tensor<512xf32>) -> ()
}
// CHECK-LABEL: func @simple_elementwise

// -----

func @invalid_matmul_bias(%arg0: !ng.tensor<512x256xf32>, %arg1: !ng.tensor<256x256xf32>) -> !ng.tensor<512x256xf32> {
  %0 = "ng.dot"(%arg0, %arg1) : (!ng.tensor<512x256xf32>, !ng.tensor<256x256xf32>) -> !ng.tensor<512x256xf32>
  %1 = "ng.add"(%arg0, %arg0) : (!ng.tensor<512x256xf32>, !ng.tensor<512x256xf32>) -> !ng.tensor<512x256xf32>
  "ng.return"(%1) : (!ng.tensor<512x256xf32>) -> ()
}
// CHECK-LABEL: func @invalid_matmul_bias

// -----

func @invalid_elementwise(%arg0: !ng.tensor<512xf32>, %arg1: !ng.tensor<512xf32>) -> !ng.tensor<512xf32> {
  %0 = "ng.mul"(%arg0, %arg1) : (!ng.tensor<512xf32>, !ng.tensor<512xf32>) -> !ng.tensor<512xf32>
  %1 = "ng.add"(%arg0, %arg1) : (!ng.tensor<512xf32>, !ng.tensor<512xf32>) -> !ng.tensor<512xf32>
  %2 = "ng.relu"(%0) : (!ng.tensor<512xf32>) -> !ng.tensor<512xf32>
  "ng.return"(%2) : (!ng.tensor<512xf32>) -> ()
}
// CHECK-LABEL: func @invalid_elementwise
