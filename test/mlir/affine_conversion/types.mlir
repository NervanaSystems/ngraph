// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// These tests verify that we can parse nGraph dialect types and lower them to affine.

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
func @i8(%arg0: !ng.i8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i16
// CHECK-SAME: (%{{.*}}: i16)
func @i16(%arg0: !ng.i16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i32
// CHECK-SAME: (%{{.*}}: i32)
func @i32(%arg0: !ng.i32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @i64
// CHECK-SAME: (%{{.*}}: i64)
func @i64(%arg0: !ng.i64) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u8
// CHECK-SAME: (%{{.*}}: i8)
func @u8(%arg0: !ng.u8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u16
// CHECK-SAME: (%{{.*}}: i16)
func @u16(%arg0: !ng.u16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @u32
// CHECK-SAME: (%{{.*}}: i32)
func @u32(%arg0: !ng.u32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @u64
// CHECK-SAME (%{{.*}}: i64)
func @u64(%arg0: !ng.u64) {
  "ng.return"() : () -> ()
}

