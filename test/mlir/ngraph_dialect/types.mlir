// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// These tests verify that we can parse nGraph dialect types and lower them to affine.

// -----

// CHECK: func @f32(%{{.*}}: f32)
func @f32(%arg0: f32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @f64(%{{.*}}: f64)
func @f64(%arg0: f64) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @i8(%{{.*}}: i8)
func @i8(%arg0: !ng.i8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @i16(%{{.*}}: i16)
func @i16(%arg0: !ng.i16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @i32(%{{.*}}: i32)
func @i32(%arg0: !ng.i32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @i64(%{{.*}}: i64)
func @i64(%arg0: !ng.i64) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @u8(%{{.*}}: i8)
func @u8(%arg0: !ng.u8) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @u16(%{{.*}}: i16)
func @u16(%arg0: !ng.u16) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @u32(%{{.*}}: i32)
func @u32(%arg0: !ng.u32) {
  "ng.return"() : () -> ()
}

// -----

// CHECK: func @u64(%{{.*}}: i64)
func @u64(%arg0: !ng.u64) {
  "ng.return"() : () -> ()
}

