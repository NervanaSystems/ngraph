// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @test()
func @test() {
  llvm.return
}
