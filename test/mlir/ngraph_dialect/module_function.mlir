// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s

// These tests verify basic functionality for nGraph module and function.

// -----

// CHECK: module
// CHECK: func @empty_func()
// CHECK: return
module {
  func @empty_func() -> () {
    "ng.return"() : () -> ()
  }
}

// -----

// CHECK: module
// CHECK: func @empty_func1()
// CHECK: return
// CHECK: func @empty_func2()
// CHECK: return
module {
  func @empty_func1() -> () {
    "ng.return"() : () -> ()
  }

  func @empty_func2() -> () {
    "ng.return"() : () -> ()
  }
}

// -----

// Empty module must be automatically generated.
// CHECK: module
// CHECK: func @no_module()
// CHECK: return
func @no_module() -> () {
  "ng.return"() : () -> ()
}

