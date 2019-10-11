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
