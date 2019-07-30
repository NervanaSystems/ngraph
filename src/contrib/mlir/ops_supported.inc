// List of all ops supported by MLIR backend end-to-end
#ifndef MLIR_OP
#define MLIR_OP
#endif

MLIR_OP(Add)
MLIR_OP(ArgMin)
MLIR_OP(ArgMax)
MLIR_OP(Divide)
MLIR_OP(Dot)
MLIR_OP(Concat)
MLIR_OP(Gather)
MLIR_OP(Greater)
MLIR_OP(Less)
MLIR_OP(Maximum)
MLIR_OP(Minimum)
MLIR_OP(Multiply)
MLIR_OP(Subtract)
MLIR_OP(Relu)
// Add new supported ops here

#undef MLIR_OP
