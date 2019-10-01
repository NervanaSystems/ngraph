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
// CHECK-SAME: (%{{.*}}: !ng.i8)
func @i8(%arg0: !ng.i8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i16
// CHECK-SAME: (%{{.*}}: !ng.i16)
func @i16(%arg0: !ng.i16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i32
// CHECK-SAME: (%{{.*}}: !ng.i32)
func @i32(%arg0: !ng.i32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i64
// CHECK-SAME: (%{{.*}}: !ng.i64)
func @i64(%arg0: !ng.i64) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u8
// CHECK-SAME: (%{{.*}}: !ng.i8)
func @u8(%arg0: !ng.u8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u16
// CHECK-SAME: (%{{.*}}: !ng.i16)
func @u16(%arg0: !ng.u16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u32
// CHECK-SAME: (%{{.*}}: !ng.i32)
func @u32(%arg0: !ng.u32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @u64
// CHECK-SAME (%{{.*}}: !ng.i64)
func @u64(%arg0: !ng.u64) {
  "ng.return"() : () -> ()
}
