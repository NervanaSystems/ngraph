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

// -----

// Convolution
// CHECK-LABEL: @convolution
// Initialization loops
// CHECK:         affine.for
// CHECK-NEXT:    affine.for
// CHECK-NEXT:    affine.for
// CHECK-NEXT:    affine.for
// CHECK:         affine.store
// Convolution loops
// CHECK:         affine.for %[[a3:.*]] = 0 to 1
// CHECK:         affine.for %[[a4:.*]] = 0 to 2
// CHECK:         affine.for %[[a5:.*]] = 0 to 2
// CHECK:         affine.for %[[a6:.*]] = 0 to 2
// CHECK:         affine.for %[[a7:.*]] = 0 to 2
// CHECK:         affine.for %[[a8:.*]] = 0 to 1
// CHECK:         affine.for %[[a9:.*]] = 0 to 1
// CHECK:         affine.load %{{.*}}[%[[a4]], %{{.*}}, %[[a8]], %[[a9]]] : memref<2x2x1x1xf32>
// CHECK:         affine.load %{{.*}}[%[[a3]], %[[a5]], %{{.*}}, {{.*}}] : memref<1x2x2x2xf32>
// CHECK-NEXT:    mulf
// CHECK-NEXT:    affine.load %{{.*}}[%[[a3]], %[[a4]], %[[a6]], %[[a7]]] : memref<1x2x2x2xf32>
// CHECK-NEXT:    %[[v4:.*]] = addf
// CHECK-NEXT:    affine.store %[[v4]], %{{.*}}[%[[a3]], %[[a4]], %[[a6]], %[[a7]]] : memref<1x2x2x2xf32>

func @convolution(%arg0: !ng.tensor<1x2x2x2xf32>, %arg1: !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32> {
  %0 = "ng.convolution"(%arg0, %arg1) {padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1]} : (!ng.tensor<1x2x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2x2xf32>) -> ()
}

// -----
//
// Group Convolution
// CHECK-DAG: #[[M0:.*]] = (d0) -> (d0 * 2)
// CHECK-DAG: #[[M1:.*]] = (d0) -> (d0 * 2 + 2)
// CHECK-DAG: #[[M2:.*]] = (d0) -> (d0)
// CHECK-DAG: #[[M3:.*]] = (d0) -> (d0 + 1)
// CHECK-DAG: #[[M8:.*]] = (d0, d1) -> (d0 + d1)
// CHECK-DAG: #[[M9:.*]] = (d0, d1) -> (d0 - d1 * 2)
// CHECK-LABEL: @groupConv
//
// Outer groups loops
// CHECK-DAG: %[[c0:.*]] = constant 0 : index
// CHECK-DAG: %[[c1:.*]] = constant 1 : index
// CHECK-DAG: %[[c2:.*]] = constant 2 : index
// CHECK:     "loop.for"(%[[c0]], %[[c2]], %[[c1]])
// CHECK-NEXT: bb0(%[[gid:.*]]: index)
//
// CHECK:     %[[v0:.*]] = affine.apply #[[M0]](%[[gid]])
// CHECK:     %[[v1:.*]] = affine.apply #[[M1]](%[[gid]])
// CHECK:     %[[v2:.*]] = affine.apply #[[M2]](%[[gid]])
// CHECK:     %[[v3:.*]] = affine.apply #[[M3]](%[[gid]])
//
// Initialization loops
// CHECK: affine.for
// CHECK-NEXT: affine.for
// CHECK-NEXT: affine.for
// CHECK-NEXT: affine.for
// %[[cst:.*]] = constant 0
// affine.store %[[cst]]
//
// Convolution loops
// CHECK:         affine.for %[[a4:.*]] = 0 to 1
// CHECK:         affine.for %[[a5:.*]] = #[[M2]](%[[v2]]) to #[[M2]](%[[v3]])
// CHECK:         affine.for %[[a6:.*]] = #[[M2]](%[[v0]]) to #[[M2]](%[[v1]])
// CHECK:         affine.for %[[a7:.*]] = 0 to 2
// CHECK:         affine.for %[[a8:.*]] = 0 to 2
// CHECK:         affine.for %[[a9:.*]] = 0 to 1
// CHECK:         affine.for %[[a10:.*]] = 0 to 1
// CHECK:         %[[v6:.*]] = affine.apply #[[M8]](%[[a7]], %[[a9]])
// CHECK:         %[[v7:.*]] = affine.apply #[[M8]](%[[a8]], %[[a10]])
// CHECK:         %[[v8:.*]] = affine.apply #[[M9]](%[[a6]], %[[a3]])
// CHECK:         affine.load %{{.*}}[%[[a5]], %[[v8]], %[[a9]], %[[a10]]] : memref<2x2x1x1xf32>
// CHECK:         affine.load %{{.*}}[%[[a4]], %[[a6]], %[[v6]], %[[v7]]] : memref<1x4x2x2xf32>
// CHECK-NEXT:    mulf
// CHECK-NEXT:    affine.load %{{.*}}[%[[a4]], %[[a5]], %[[a7]], %[[a8]]] : memref<1x2x2x2xf32>
// CHECK-NEXT:    %[[v4:.*]] = addf
// CHECK-NEXT:    affine.store %[[v4]], %{{.*}}[%[[a4]], %[[a5]], %[[a7]], %[[a8]]] : memref<1x2x2x2xf32>

func @groupConv(%arg0: !ng.tensor<1x4x2x2xf32>, %arg1: !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32> {
  %0 = "ng.groupConv"(%arg0, %arg1) {groups = 2 : i64, padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1]} : (!ng.tensor<1x4x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2x2xf32>) -> ()
}
