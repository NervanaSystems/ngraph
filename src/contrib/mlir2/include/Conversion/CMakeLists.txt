set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(MLIRConversionPassIncGen)

message(STATUS "*********** pre-tablegen passes")
add_mlir_doc(Passes -gen-pass-doc ConversionPasses ./)
message(STATUS "output ${TABLEGEN_OUTPUT}")
message(STATUS "*********** post-tablegen passes")
