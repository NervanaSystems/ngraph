// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// Verify that core operations are properly converted to affine dialect.

// -----

// Gather Op
// CHECK: affine.for %[[I:.*]] = 0 to 16 {
// CHECK:   %[[L0:.*]] = affine.load %{{.*}}[%[[I]]]
// CHECK:   %[[GATHER_IDX:.*]] = index_cast %[[L0]]
// CHECK:   affine.for %[[J:.*]] = 0 to 32 {
// CHECK:     %[[VALUE:.*]] = load %{{.*}}[%[[GATHER_IDX]], %[[J]]]
// CHECK:     affine.store %[[VALUE]], {{.*}}[%[[I]], %[[J]]]
func @simple_gather(%arg0: !ng.tensor<16x!ng.i64>, %arg1: !ng.tensor<512x32xf32>) -> !ng.tensor<16x32xf32> {
   %0 = "ng.gather"(%arg1, %arg0) {axis = 0 : i64} : (!ng.tensor<512x32xf32>, !ng.tensor<16x!ng.i64>) -> !ng.tensor<16x32xf32>
  "ng.return"(%0) : (!ng.tensor<16x32xf32>) -> ()
}

// -----

// Dot Op
// CHECK:       affine.for %[[I:.*]] = 0 to 16
// CHECK-NEXT:  affine.for %[[J:.*]] = 0 to 32
// CHECK-NEXT:  affine.store %{{.*}}, %[[RESULT:.*]][%[[I]], %[[J]]]
// CHECK:       }
// CHECK-NEXT:  }
// CHECK:       affine.for %[[K:.*]] = 0 to 16
// CHECK-NEXT:  affine.for {{%.*}} = 0 to 8
// CHECK-NEXT:  affine.for %[[M:.*]] = 0 to 32
// CHECK:       affine.load
// CHECK:       affine.load
// CHECK:       mulf
// CHECK:       %[[R:.*]] = addf 
// CHECK:       affine.store %[[R]], %[[RESULT]][%[[K]], %[[M]]]
func @simple_dot(%arg0: !ng.tensor<16x8xf32>, %arg1: !ng.tensor<8x32xf32>) -> !ng.tensor<16x32xf32> {
   %0 = "ng.dot"(%arg0, %arg1) : (!ng.tensor<16x8xf32>, !ng.tensor<8x32xf32>) -> !ng.tensor<16x32xf32>
  "ng.return"(%0) : (!ng.tensor<16x32xf32>) -> ()
}

// -----

// std.view 

// CHECK-DAG: #[[MAP0:[a-zA-Z0-9]+]] = (d0, d1) -> (d0 * 2 + d1)
// CHECK:       %[[T1:[0-9]+]] = alloc() : memref<24xi8>
// CHECK-NEXT:  %[[T2:[0-9]+]] = std.view %[[T1]][][] : memref<24xi8> to memref<3x2xf32, #[[MAP0]]>
// CHECK:       affine.store %{{[0-9]+}}, %[[T2]][%{{.*}}, %{{.*}}] : memref<3x2xf32, #[[MAP0]]>
//
// CHECK:       %[[T4:[0-9]+]] = std.view %[[T1]][][] : memref<24xi8> to memref<3x2xf32, #[[MAP0]]>
// CHECK:       affine.store %{{[0-9]+}}, %[[T4]][%{{.*}}, %{{.*}}] : memref<3x2xf32, #[[MAP0]]>

func @add(%arg0: !ng.tensor<3x2xf32>, %arg1: !ng.tensor<3x2xf32>) -> !ng.tensor<3x2xf32> {
  %0 = "ng.add"(%arg0, %arg1) {ng.buffer_id = 0 : i64} : (!ng.tensor<3x2xf32>, !ng.tensor<3x2xf32>) -> !ng.tensor<3x2xf32>
  %2 = "ng.add"(%0, %0) {ng.buffer_id = 0 : i64}: (!ng.tensor<3x2xf32>, !ng.tensor<3x2xf32>) -> !ng.tensor<3x2xf32>
  %3 = "ng.add"(%2, %2) : (!ng.tensor<3x2xf32>, !ng.tensor<3x2xf32>) -> !ng.tensor<3x2xf32>
  "ng.return"(%3) : (!ng.tensor<3x2xf32>) -> ()
}
