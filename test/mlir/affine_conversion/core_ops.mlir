// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// Verify that core operations are properly converted to affine dialect.

// -----

// Gather Op
// CHECK: affine.for [[i:%.*]] = 0 to 16 {
// CHECK:   [[L0:%.*]] = affine.load %{{.*\[}}[[i]]{{\]}}
// CHECK:   [[GATHER_IDX:%.*]] = index_cast [[L0]]
// CHECK:   affine.for [[j:%.*]] = 0 to 32 {
// CHECK:     [[VALUE:%.*]] = load %{{.*\[}}[[GATHER_IDX]], [[j]]{{\]}}
// CHECK:     affine.store [[VALUE]], %{{.*\[}}[[i]], [[j]]{{\]}}
func @simple_gather(%arg0: !ng.tensor<16x!ng.i64>, %arg1: !ng.tensor<512x32xf32>) -> !ng.tensor<16x32xf32> {
   %0 = "ng.gather"(%arg1, %arg0) {axis = 0 : i64} : (!ng.tensor<512x32xf32>, !ng.tensor<16x!ng.i64>) -> !ng.tensor<16x32xf32>
  "ng.return"(%0) : (!ng.tensor<16x32xf32>) -> ()
}
