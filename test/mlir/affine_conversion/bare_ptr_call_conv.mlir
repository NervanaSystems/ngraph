// RUN: ngraph-opt %s -convert-ngraph-to-affine -ngraph-bare-ptr-memref-lowering -split-input-file | FileCheck %s --check-prefix=BARE-PTR-CC
// RUN: ngraph-opt %s -convert-ngraph-to-affine -split-input-file | FileCheck %s --check-prefix=STD-CC

// Tests related to the bare pointer calling convention.

// Verify that the `noalias` attribute is generated when the bare pointer calling
// convention is used but not with the standard calling convention.
func @noalias_attribute(%arg0: !ng.tensor<16x!ng.i64>, %arg1: !ng.tensor<512x32xf32>){
  "ng.return"() : () -> ()
}
// BARE-PTR-CC-LABEL: func @noalias_attribute
// BARE-PTR-CC-SAME:  %{{.*}}: memref<16xi64> {llvm.noalias = true}
// BARE-PTR-CC-SAME:  %{{.*}}: memref<512x32xf32> {llvm.noalias = true})

// STD-CC-LABEL: func @noalias_attribute
// STD-CC-NOT:  llvm.noalias
