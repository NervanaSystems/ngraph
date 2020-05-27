// RUN: ngraph-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: ngraph-opt %s -split-input-file | ngraph-opt | FileCheck %s

// These tests verify parsing and printing of nGraph types.

// -----

// CHECK-LABEL: func @f32
// CHECK-SAME: (%{{.*}}: f32)
func @f32(%arg0: f32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @f64
// CHECK-SAME: (%{{.*}}: f64)
func @f64(%arg0: f64) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i8
// CHECK-SAME: (%{{.*}}: i8)
func @i8(%arg0: i8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i16
// CHECK-SAME: (%{{.*}}: i16)
func @i16(%arg0: i16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i32
// CHECK-SAME: (%{{.*}}: i32)
func @i32(%arg0: i32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i64
// CHECK-SAME: (%{{.*}}: i64)
func @i64(%arg0: i64) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u8
// CHECK-SAME: (%{{.*}}: ui8)
func @u8(%arg0: ui8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u16
// CHECK-SAME: (%{{.*}}: ui16)
func @u16(%arg0: ui16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u32
// CHECK-SAME: (%{{.*}}: ui32)
func @u32(%arg0: ui32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @u64
// CHECK-SAME (%{{.*}}: i64)
func @u64(%arg0: ui64) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_i8
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xi8>)
func @tensor_i8(%arg0: !ng.tensor<2x2xi8>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_i16
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xi16>)
func @tensor_i16(%arg0: !ng.tensor<2x2xi16>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_i32
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xi32>)
func @tensor_i32(%arg0: !ng.tensor<2x2xi32>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_i64
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xi64>)
func @tensor_i64(%arg0: !ng.tensor<2x2xi64>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_u8
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xui8>)
func @tensor_u8(%arg0: !ng.tensor<2x2xui8>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_u16
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xui16>)
func @tensor_u16(%arg0: !ng.tensor<2x2xui16>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_u32
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xui32>)
func @tensor_u32(%arg0: !ng.tensor<2x2xui32>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_u64
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xui64>)
func @tensor_u64(%arg0: !ng.tensor<2x2xui64>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_f32
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xf32>)
func @tensor_f32(%arg0: !ng.tensor<2x2xf32>) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @tensor_f64
// CHECK-SAME: (%{{.*}}: !ng.tensor<2x2xf64>)
func @tensor_f64(%arg0: !ng.tensor<2x2xf64>) {
  "ng.return"() : () -> ()
}
