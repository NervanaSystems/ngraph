// RUN: ngraph-opt %s -convert-ngraph-to-affine | FileCheck %s

// These tests verify basic functionality for nGraph module and function.

// CHECK-LABEL: func @empty_func
// CHECK: module
// CHECK: func @empty_func()
// CHECK: return
module {
  func @empty_func() -> () {
    "ng.return"() : () -> ()
  }
}

