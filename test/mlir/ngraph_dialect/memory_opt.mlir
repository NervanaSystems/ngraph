// RUN: ngraph-opt %s -split-input-file -ng-memory-opt | FileCheck %s


// func @main(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
//   %0 = "ng.add"(%arg0, %arg1) {ng.buffer_id = 0 : i32, ng.buffer_offset = 0 : i32} : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//   %1 = "ng.add"(%0, %0) {ng.buffer_id = 0 : i32, ng.buffer_offset = 0 : i32} : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//   %2 = "ng.add"(%1, %1) {ng.buffer_id = 0 : i32, ng.buffer_offset = 0 : i32} : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//   %3 = "ng.add"(%2, %2) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//   "ng.return"(%3) : (!ng.tensor<2x2xf32>) -> ()
// }

// CHECK:      buffer_id = 0
// CHECK-SAME: buffer_offset = 0
//
// CHECK-NEXT: buffer_id = 0
// CHECK-SAME: buffer_offset = 0
//
// CHECK-NEXT: buffer_id = 0
// CHECK-SAME: buffer_offset = 0
// CHECK-NOT:  buffer_id
// CHECK-NOT:  buffer_offset
// CHECK:      return
//func @main(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
//  %0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %1 = "ng.add"(%0, %0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %2 = "ng.add"(%1, %1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  %3 = "ng.add"(%2, %2) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
//  "ng.return"(%3) : (!ng.tensor<2x2xf32>) -> ()
//}

// -----

func @main(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<4x2xf32> {
  %0 = "ng.add"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %1 = "ng.add"(%0, %0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
  %2 = "ng.concat"(%0, %1) {concatenation_axis = 0} : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<4x2xf32>
  %3 = "ng.add"(%2, %2) : (!ng.tensor<4x2xf32>, !ng.tensor<4x2xf32>) -> !ng.tensor<4x2xf32>
  "ng.return"(%3) : (!ng.tensor<4x2xf32>) -> ()
}