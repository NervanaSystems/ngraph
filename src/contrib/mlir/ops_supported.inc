// List of all ops supported by MLIR backend end-to-end
#ifndef MLIR_OP
#define MLIR_OP
#endif

MLIR_OP(Add)
MLIR_OP(ArgMin)
MLIR_OP(ArgMax)
MLIR_OP(Relu)
MLIR_OP(Dot)
// Add new supported ops here

#undef MLIR_OP
