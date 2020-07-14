// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// Verify that core operations are properly converted to affine dialect.

// -----

// Gather Op
// CHECK-LABEL: func @simple_gather
//       CHECK: affine.for %[[I:.*]] = 0 to 16 {
//       CHECK:   %[[L0:.*]] = affine.load %{{.*}}[%[[I]]]
//       CHECK:   %[[GATHER_IDX:.*]] = index_cast %[[L0]]
//       CHECK:   affine.for %[[J:.*]] = 0 to 32 {
//       CHECK:     %[[VALUE:.*]] = load %{{.*}}[%[[GATHER_IDX]], %[[J]]]
//       CHECK:     affine.store %[[VALUE]], {{.*}}[%[[I]], %[[J]]]
func @simple_gather(%arg0: !ng.tensor<16x!ng.i64>, %arg1: !ng.tensor<512x32xf32>) -> !ng.tensor<16x32xf32> {
   %0 = "ng.gather"(%arg1, %arg0) {axis = 0 : i64} : (!ng.tensor<512x32xf32>, !ng.tensor<16x!ng.i64>) -> !ng.tensor<16x32xf32>
  "ng.return"(%0) : (!ng.tensor<16x32xf32>) -> ()
}

// -----

// Equal Op
// CHECK-LABEL: func @simple_equal
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "oeq", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_equal(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.equal"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// -----

// NotEqual Op
// CHECK-LABEL: func @simple_notequal
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "one", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_notequal(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.not.equal"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// -----

// Greater Op
// CHECK-LABEL: func @simple_greater
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "ogt", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_greater(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.greater"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// Greater Op (signed int)
// CHECK-LABEL: func @simple_greater_int
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "sgt", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_greater_int(%arg0: !ng.tensor<2x2xi32>, %arg1: !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.greater"(%arg1, %arg0) : (!ng.tensor<2x2xi32>, !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// Greater Op (unsigned int)
// CHECK-LABEL: func @simple_greater_uint
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "ugt", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_greater_uint(%arg0: !ng.tensor<2x2xui32>, %arg1: !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.greater"(%arg1, %arg0) : (!ng.tensor<2x2xui32>, !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// -----

// GreaterEq Op
// CHECK-LABEL: func @simple_greatereq
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "oge", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_greatereq(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.greater.eq"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// GreaterEq Op (signed int)
// CHECK-LABEL: func @simple_greatereq_int
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "sge", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_greatereq_int(%arg0: !ng.tensor<2x2xi32>, %arg1: !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.greater.eq"(%arg1, %arg0) : (!ng.tensor<2x2xi32>, !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// GreaterEq Op (unsigned int)
// CHECK-LABEL: func @simple_greatereq_uint
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "uge", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_greatereq_uint(%arg0: !ng.tensor<2x2xui32>, %arg1: !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.greater.eq"(%arg1, %arg0) : (!ng.tensor<2x2xui32>, !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// -----

// Less Op
// CHECK-LABEL: func @simple_less
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "olt", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_less(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.less"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// Less Op (signed int)
// CHECK-LABEL: func @simple_less_int
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "slt", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_less_int(%arg0: !ng.tensor<2x2xi32>, %arg1: !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.less"(%arg1, %arg0) : (!ng.tensor<2x2xi32>, !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// Less Op (unsigned int)
// CHECK-LABEL: func @simple_less_uint
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "ult", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_less_uint(%arg0: !ng.tensor<2x2xui32>, %arg1: !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.less"(%arg1, %arg0) : (!ng.tensor<2x2xui32>, !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// -----

// LessEq Op
// CHECK-LABEL: func @simple_lesseq
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "ole", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_lesseq(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.less.eq"(%arg1, %arg0) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// LessEq Op (signed int)
// CHECK-LABEL: func @simple_lesseq_int
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "sle", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_lesseq_int(%arg0: !ng.tensor<2x2xi32>, %arg1: !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.less.eq"(%arg1, %arg0) : (!ng.tensor<2x2xi32>, !ng.tensor<2x2xi32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// LessEq Op (unsigned int)
// CHECK-LABEL: func @simple_lesseq_uint
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ONE:.*]] = constant 1 : i8 
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i8 
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "ule", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[ONE]], %[[ZERO]] : i8
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_lesseq_uint(%arg0: !ng.tensor<2x2xui32>, %arg1: !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>{
%0 = "ng.less.eq"(%arg1, %arg0) : (!ng.tensor<2x2xui32>, !ng.tensor<2x2xui32>) -> !ng.tensor<2x2x!ng.u8>
   "ng.return"(%0) : (!ng.tensor<2x2x!ng.u8>) -> ()
}

// -----

// Max Op
// CHECK-LABEL: func @simple_max
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "ogt", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_max(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
   %0 = "ng.max"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
   "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// Max Op (signed int)
// CHECK-LABEL: func @simple_max_int
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "sgt", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_max_int(%arg0: !ng.tensor<2x2xi32>, %arg1: !ng.tensor<2x2xi32>) -> !ng.tensor<2x2xi32> {
   %0 = "ng.max"(%arg0, %arg1) : (!ng.tensor<2x2xi32>, !ng.tensor<2x2xi32>) -> !ng.tensor<2x2xi32>
   "ng.return"(%0) : (!ng.tensor<2x2xi32>) -> ()
}

// Max Op (unsigned int)
// CHECK-LABEL: func @simple_max_uint
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "ugt", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_max_uint(%arg0: !ng.tensor<2x2xui32>, %arg1: !ng.tensor<2x2xui32>) -> !ng.tensor<2x2xui32> {
   %0 = "ng.max"(%arg0, %arg1) : (!ng.tensor<2x2xui32>, !ng.tensor<2x2xui32>) -> !ng.tensor<2x2xui32>
   "ng.return"(%0) : (!ng.tensor<2x2xui32>) -> ()
}

// -----

// Min Op
// CHECK-LABEL: func @simple_min
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "olt", %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : f32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_min(%arg0: !ng.tensor<2x2xf32>, %arg1: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
   %0 = "ng.min"(%arg0, %arg1) : (!ng.tensor<2x2xf32>, !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
   "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// Min Op (signed int)
// CHECK-LABEL: func @simple_min_int
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "slt", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_min_int(%arg0: !ng.tensor<2x2xi32>, %arg1: !ng.tensor<2x2xi32>) -> !ng.tensor<2x2xi32> {
   %0 = "ng.min"(%arg0, %arg1) : (!ng.tensor<2x2xi32>, !ng.tensor<2x2xi32>) -> !ng.tensor<2x2xi32>
   "ng.return"(%0) : (!ng.tensor<2x2xi32>) -> ()
}

// Min Op (unsigned int)
// CHECK-LABEL: func @simple_min_uint
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[LHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[RHS:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "ult", %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_min_uint(%arg0: !ng.tensor<2x2xui32>, %arg1: !ng.tensor<2x2xui32>) -> !ng.tensor<2x2xui32> {
   %0 = "ng.min"(%arg0, %arg1) : (!ng.tensor<2x2xui32>, !ng.tensor<2x2xui32>) -> !ng.tensor<2x2xui32>
   "ng.return"(%0) : (!ng.tensor<2x2xui32>) -> ()
}

// -----

// Relu Op
// CHECK-LABEL: func @simple_relu
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[DATA:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xf32>
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0.{{0*}}e+00 : f32
// CHECK-NEXT:      %[[CMP:.*]] = cmpf "ogt", %[[DATA]], %[[ZERO]] : f32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[DATA]], %[[ZERO]] : f32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_relu(%arg0: !ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32> {
   %0 = "ng.relu"(%arg0) : (!ng.tensor<2x2xf32>) -> !ng.tensor<2x2xf32>
   "ng.return"(%0) : (!ng.tensor<2x2xf32>) -> ()
}

// Relu Op (signed int)
// CHECK-LABEL: func @simple_relu_int
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[DATA:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i32
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "sgt", %[[DATA]], %[[ZERO]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[DATA]], %[[ZERO]] : i32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_relu_int(%arg0: !ng.tensor<2x2xi32>) -> !ng.tensor<2x2xi32> {
   %0 = "ng.relu"(%arg0) : (!ng.tensor<2x2xi32>) -> !ng.tensor<2x2xi32>
   "ng.return"(%0) : (!ng.tensor<2x2xi32>) -> ()
}

// Relu Op (unsigned int)
// CHECK-LABEL: func @simple_relu_uint
//      CHECK:  affine.for %[[I:.*]] = 0 to 2
// CHECK-NEXT:    affine.for %[[J:.*]] = 0 to 2
// CHECK-NEXT:      %[[DATA:.*]] = affine.load  %{{.*}}[%[[I]], %[[J]]] : memref<2x2xi32>
// CHECK-NEXT:      %[[ZERO:.*]] = constant 0 : i32
// CHECK-NEXT:      %[[CMP:.*]] = cmpi "ugt", %[[DATA]], %[[ZERO]] : i32
// CHECK-NEXT:      %[[RES:.*]] = select %[[CMP]], %[[DATA]], %[[ZERO]] : i32
// CHECK-NEXT:      affine.store %[[RES]], %{{.*}}[%[[I]], %[[J]]]
func @simple_relu_uint(%arg0: !ng.tensor<2x2xui32>) -> !ng.tensor<2x2xui32> {
   %0 = "ng.relu"(%arg0) : (!ng.tensor<2x2xui32>) -> !ng.tensor<2x2xui32>
   "ng.return"(%0) : (!ng.tensor<2x2xui32>) -> ()
}

// -----

// Dot Op
// CHECK-LABEL: func @simple_dot
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

// CHECK:     #[[MAP0:[a-zA-Z0-9]+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
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
// CHECK-LABEL: func @convolution
// Initialization loops
// CHECK:         affine.for
// CHECK-NEXT:      affine.for
// CHECK-NEXT:        affine.for
// CHECK-NEXT:          affine.for
// CHECK:                 affine.store
// Convolution loops
// CHECK:         affine.for %[[a3:.*]] = 0 to 1
// CHECK:           affine.for %[[a4:.*]] = 0 to 2
// CHECK:             affine.for %[[a5:.*]] = 0 to 2
// CHECK:               affine.for %[[a6:.*]] = 0 to 2
// CHECK:                 affine.for %[[a7:.*]] = 0 to 2
// CHECK:                   affine.for %[[a8:.*]] = 0 to 1
// CHECK:                     affine.for %[[a9:.*]] = 0 to 1
// CHECK:                       affine.load %{{.*}}[%[[a4]], %{{.*}}, %[[a8]], %[[a9]]] : memref<2x2x1x1xf32>
// CHECK:                       affine.load %{{.*}}[%[[a3]], %[[a5]], %{{.*}}, {{.*}}] : memref<1x2x2x2xf32>
// CHECK-NEXT:                  mulf
// CHECK-NEXT:                  affine.load %{{.*}}[%[[a3]], %[[a4]], %[[a6]], %[[a7]]] : memref<1x2x2x2xf32>
// CHECK-NEXT:                  %[[v4:.*]] = addf
// CHECK-NEXT:                  affine.store %[[v4]], %{{.*}}[%[[a3]], %[[a4]], %[[a6]], %[[a7]]] : memref<1x2x2x2xf32>

func @convolution(%arg0: !ng.tensor<1x2x2x2xf32>, %arg1: !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32> {
  %0 = "ng.convolution"(%arg0, %arg1) {padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1]} : (!ng.tensor<1x2x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2x2xf32>) -> ()
}

// -----
//
// Group Convolution
// CHECK-DAG: #[[M0:.*]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG: #[[M1:.*]] = affine_map<(d0) -> (d0 * 2 + 2)>
// CHECK-DAG: #[[M2:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[M3:.*]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: #[[M8:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #[[M9:.*]] = affine_map<(d0, d1) -> (d0 - d1 * 2)>
// CHECK-LABEL: func @groupConv
//
// Outer groups loops
// CHECK:      affine.for %[[gid:.*]] = 0 to 2
// CHECK:     %[[v0:.*]] = affine.apply #[[M0]](%[[gid]])
// CHECK:     %[[v1:.*]] = affine.apply #[[M1]](%[[gid]])
// CHECK:     %[[v2:.*]] = affine.apply #[[M2]](%[[gid]])
// CHECK:     %[[v3:.*]] = affine.apply #[[M3]](%[[gid]])
//
// Initialization loops
// CHECK:       affine.for
// CHECK-NEXT:    affine.for
// CHECK-NEXT:      affine.for
// CHECK-NEXT:        affine.for
// CHECK:               %[[cst:.*]] = constant 0
// CHECK:               affine.store %[[cst]]
//
// Convolution loops
// CHECK:         affine.for %[[a4:.*]] = 0 to 1
// CHECK:           affine.for %[[a5:.*]] = #[[M2]](%[[v2]]) to #[[M2]](%[[v3]])
// CHECK:             affine.for %[[a6:.*]] = #[[M2]](%[[v0]]) to #[[M2]](%[[v1]])
// CHECK:               affine.for %[[a7:.*]] = 0 to 2
// CHECK:                 affine.for %[[a8:.*]] = 0 to 2
// CHECK:                   affine.for %[[a9:.*]] = 0 to 1
// CHECK:                     affine.for %[[a10:.*]] = 0 to 1
// CHECK:                       %[[v6:.*]] = affine.apply #[[M8]](%[[a7]], %[[a9]])
// CHECK:                       %[[v7:.*]] = affine.apply #[[M8]](%[[a8]], %[[a10]])
// CHECK:                       %[[v8:.*]] = affine.apply #[[M9]](%[[a6]], %[[a3]])
// CHECK:                       affine.load %{{.*}}[%[[a5]], %[[v8]], %[[a9]], %[[a10]]] : memref<2x2x1x1xf32>
// CHECK:                       affine.load %{{.*}}[%[[a4]], %[[a6]], %[[v6]], %[[v7]]] : memref<1x4x2x2xf32>
// CHECK-NEXT:                  mulf
// CHECK-NEXT:                  affine.load %{{.*}}[%[[a4]], %[[a5]], %[[a7]], %[[a8]]] : memref<1x2x2x2xf32>
// CHECK-NEXT:                  %[[v4:.*]] = addf
// CHECK-NEXT:                  affine.store %[[v4]], %{{.*}}[%[[a4]], %[[a5]], %[[a7]], %[[a8]]] : memref<1x2x2x2xf32>

func @groupConv(%arg0: !ng.tensor<1x4x2x2xf32>, %arg1: !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32> {
  %0 = "ng.groupConv"(%arg0, %arg1) {groups = 2 : i64, padAbove = [0, 0], padBelow = [0, 0], strides = [1, 1]} : (!ng.tensor<1x4x2x2xf32>, !ng.tensor<2x2x1x1xf32>) -> !ng.tensor<1x2x2x2xf32>
  "ng.return"(%0) : (!ng.tensor<1x2x2x2xf32>) -> ()
}


// -----

// argmin
// CHECK:      func @argmin(%[[INPUT:.*]]: memref<4x3xf32>, %[[OUTPUT:.*]]: memref<3xi32>) {
// CHECK:        affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:     %[[ZERO:.*]] = index_cast %[[CONST_ZERO:.*]] : index to i32
// CHECK-NEXT:     store %[[ZERO]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %[[ROW:.*]] = 0 to 4 {
// CHECK-NEXT:     affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:       %[[MIN:.*]] = load %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:       %[[MIN_INDEX:.*]] = index_cast %[[MIN]] : i32 to index
// CHECK-NEXT:       %[[INPUT_MIN:.*]] = load %[[INPUT]][%[[MIN_INDEX]], %[[COL]]] : memref<4x3xf32>
// CHECK-NEXT:       %[[INPUT_CURR:.*]] = affine.load %[[INPUT]][%[[ROW]], %[[COL]]] : memref<4x3xf32>
// CHECK-NEXT:       %[[CMP:.*]] = cmpf "olt", %[[INPUT_CURR]], %[[INPUT_MIN]] : f32
// CHECK-NEXT:       %[[NEW_MIN_INDEX:.*]] = select %[[CMP]], %[[ROW]], %[[MIN_INDEX]] : index
// CHECK-NEXT:       %[[NEW_MIN:.*]] = index_cast %[[NEW_MIN_INDEX]] : index to i32
// CHECK-NEXT:       store %[[NEW_MIN]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
func @argmin(%arg0: !ng.tensor<4x3xf32>) -> !ng.tensor<3xsi32> {
   %0 = "ng.argmin.red"(%arg0) {axes = [0]} : (!ng.tensor<4x3xf32>) -> !ng.tensor<3xsi32>
   "ng.return"(%0) : (!ng.tensor<3xsi32>) -> ()
}

// argmin (signed int)
// CHECK:      func @argmin_int(%[[INPUT:.*]]: memref<4x3xi32>, %[[OUTPUT:.*]]: memref<3xi32>) {
// CHECK:        affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:     %[[ZERO:.*]] = index_cast %[[CONST_ZERO:.*]] : index to i32
// CHECK-NEXT:     store %[[ZERO]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %[[ROW:.*]] = 0 to 4 {
// CHECK-NEXT:     affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:       %[[MIN:.*]] = load %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:       %[[MIN_INDEX:.*]] = index_cast %[[MIN]] : i32 to index
// CHECK-NEXT:       %[[INPUT_MIN:.*]] = load %[[INPUT]][%[[MIN_INDEX]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[INPUT_CURR:.*]] = affine.load %[[INPUT]][%[[ROW]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[CMP:.*]] = cmpi "slt", %[[INPUT_CURR]], %[[INPUT_MIN]] : i32
// CHECK-NEXT:       %[[NEW_MIN_INDEX:.*]] = select %[[CMP]], %[[ROW]], %[[MIN_INDEX]] : index
// CHECK-NEXT:       %[[NEW_MIN:.*]] = index_cast %[[NEW_MIN_INDEX]] : index to i32
// CHECK-NEXT:       store %[[NEW_MIN]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
func @argmin_int(%arg0: !ng.tensor<4x3xsi32>) -> !ng.tensor<3xsi32> {
   %0 = "ng.argmin.red"(%arg0) {axes = [0]} : (!ng.tensor<4x3xsi32>) -> !ng.tensor<3xsi32>
   "ng.return"(%0) : (!ng.tensor<3xsi32>) -> ()
}

// argmin (unsigned int)
// CHECK:      func @argmin_uint(%[[INPUT:.*]]: memref<4x3xi32>, %[[OUTPUT:.*]]: memref<3xi32>) {
// CHECK:        affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:     %[[ZERO:.*]] = index_cast %[[CONST_ZERO:.*]] : index to i32
// CHECK-NEXT:     store %[[ZERO]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %[[ROW:.*]] = 0 to 4 {
// CHECK-NEXT:     affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:       %[[MIN:.*]] = load %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:       %[[MIN_INDEX:.*]] = index_cast %[[MIN]] : i32 to index
// CHECK-NEXT:       %[[INPUT_MIN:.*]] = load %[[INPUT]][%[[MIN_INDEX]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[INPUT_CURR:.*]] = affine.load %[[INPUT]][%[[ROW]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[CMP:.*]] = cmpi "ult", %[[INPUT_CURR]], %[[INPUT_MIN]] : i32
// CHECK-NEXT:       %[[NEW_MIN_INDEX:.*]] = select %[[CMP]], %[[ROW]], %[[MIN_INDEX]] : index
// CHECK-NEXT:       %[[NEW_MIN:.*]] = index_cast %[[NEW_MIN_INDEX]] : index to i32
// CHECK-NEXT:       store %[[NEW_MIN]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
func @argmin_uint(%arg0: !ng.tensor<4x3xui32>) -> !ng.tensor<3xsi32> {
   %0 = "ng.argmin.red"(%arg0) {axes = [0]} : (!ng.tensor<4x3xui32>) -> !ng.tensor<3xsi32>
   "ng.return"(%0) : (!ng.tensor<3xsi32>) -> ()
}

// -----

// argmax
// CHECK:      func @argmax(%[[INPUT:.*]]: memref<4x3xf32>, %[[OUTPUT:.*]]: memref<3xi32>) {
// CHECK:        affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:     %[[ZERO:.*]] = index_cast %[[CONST_ZERO:.*]] : index to i32
// CHECK-NEXT:     store %[[ZERO]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %[[ROW:.*]] = 0 to 4 {
// CHECK-NEXT:     affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:       %[[MAX:.*]] = load %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:       %[[MAX_INDEX:.*]] = index_cast %[[MAX]] : i32 to index
// CHECK-NEXT:       %[[INPUT_CURR:.*]] = affine.load %[[INPUT]][%[[ROW]], %[[COL]]] : memref<4x3xf32>
// CHECK-NEXT:       %[[INPUT_MAX:.*]] = load %[[INPUT]][%[[MAX_INDEX]], %[[COL]]] : memref<4x3xf32>
// CHECK-NEXT:       %[[CMP:.*]] = cmpf "olt", %[[INPUT_MAX]], %[[INPUT_CURR]] : f32
// CHECK-NEXT:       %[[NEW_MAX_INDEX:.*]] = select %[[CMP]], %[[ROW]], %[[MAX_INDEX]] : index
// CHECK-NEXT:       %[[NEW_MAX:.*]] = index_cast %[[NEW_MAX_INDEX]] : index to i32
// CHECK-NEXT:       store %[[NEW_MAX]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
func @argmax(%arg0: !ng.tensor<4x3xf32>) -> !ng.tensor<3xsi32> {
   %0 = "ng.argmax.red"(%arg0) {axes = [0]} : (!ng.tensor<4x3xf32>) -> !ng.tensor<3xsi32>
   "ng.return"(%0) : (!ng.tensor<3xsi32>) -> ()
}

// argmax (signed int)
// CHECK:      func @argmax_int(%[[INPUT:.*]]: memref<4x3xi32>, %[[OUTPUT:.*]]: memref<3xi32>) {
// CHECK:        affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:     %[[ZERO:.*]] = index_cast %[[CONST_ZERO:.*]] : index to i32
// CHECK-NEXT:     store %[[ZERO]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %[[ROW:.*]] = 0 to 4 {
// CHECK-NEXT:     affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:       %[[MAX:.*]] = load %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:       %[[MAX_INDEX:.*]] = index_cast %[[MAX]] : i32 to index
// CHECK-NEXT:       %[[INPUT_CURR:.*]] = affine.load %[[INPUT]][%[[ROW]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[INPUT_MAX:.*]] = load %[[INPUT]][%[[MAX_INDEX]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[CMP:.*]] = cmpi "slt", %[[INPUT_MAX]], %[[INPUT_CURR]] : i32
// CHECK-NEXT:       %[[NEW_MAX_INDEX:.*]] = select %[[CMP]], %[[ROW]], %[[MAX_INDEX]] : index
// CHECK-NEXT:       %[[NEW_MAX:.*]] = index_cast %[[NEW_MAX_INDEX]] : index to i32
// CHECK-NEXT:       store %[[NEW_MAX]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
func @argmax_int(%arg0: !ng.tensor<4x3xsi32>) -> !ng.tensor<3xsi32> {
   %0 = "ng.argmax.red"(%arg0) {axes = [0]} : (!ng.tensor<4x3xsi32>) -> !ng.tensor<3xsi32>
   "ng.return"(%0) : (!ng.tensor<3xsi32>) -> ()
}

// argmax (unsigned int)
// CHECK:      func @argmax_uint(%[[INPUT:.*]]: memref<4x3xi32>, %[[OUTPUT:.*]]: memref<3xi32>) {
// CHECK:        affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:     %[[ZERO:.*]] = index_cast %[[CONST_ZERO:.*]] : index to i32
// CHECK-NEXT:     store %[[ZERO]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %[[ROW:.*]] = 0 to 4 {
// CHECK-NEXT:     affine.for %[[COL:.*]] = 0 to 3 {
// CHECK-NEXT:       %[[MAX:.*]] = load %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:       %[[MAX_INDEX:.*]] = index_cast %[[MAX]] : i32 to index
// CHECK-NEXT:       %[[INPUT_CURR:.*]] = affine.load %[[INPUT]][%[[ROW]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[INPUT_MAX:.*]] = load %[[INPUT]][%[[MAX_INDEX]], %[[COL]]] : memref<4x3xi32>
// CHECK-NEXT:       %[[CMP:.*]] = cmpi "ult", %[[INPUT_MAX]], %[[INPUT_CURR]] : i32
// CHECK-NEXT:       %[[NEW_MAX_INDEX:.*]] = select %[[CMP]], %[[ROW]], %[[MAX_INDEX]] : index
// CHECK-NEXT:       %[[NEW_MAX:.*]] = index_cast %[[NEW_MAX_INDEX]] : index to i32
// CHECK-NEXT:       store %[[NEW_MAX]], %[[OUTPUT]][%[[COL]]] : memref<3xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
func @argmax_uint(%arg0: !ng.tensor<4x3xui32>) -> !ng.tensor<3xsi32> {
   %0 = "ng.argmax.red"(%arg0) {axes = [0]} : (!ng.tensor<4x3xui32>) -> !ng.tensor<3xsi32>
   "ng.return"(%0) : (!ng.tensor<3xsi32>) -> ()
}
