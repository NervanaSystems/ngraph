// RUN: ngraph-opt %s -split-input-file | FileCheck %s
// Verify the printed output can be parsed.
// RUN: ngraph-opt %s -split-input-file | ngraph-opt | FileCheck %s

// These tests verify parsing and printing of various combinations of nGraph module and function
// ops.

// -----

// CHECK: module {
// CHECK: func @empty_func() {
// CHECK: return
module {
  func @empty_func() -> () {
    "ng.return"() : () -> ()
  }
}

// -----

// CHECK: module {
// CHECK: func @empty_func1() {
// CHECK: ng.return
// CHECK: func @empty_func2() {
// CHECK: ng.return
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
// CHECK: module {
// CHECK: func @no_module() {
// CHECK: ng.return
func @no_module() -> () {
  "ng.return"() : () -> ()
}

