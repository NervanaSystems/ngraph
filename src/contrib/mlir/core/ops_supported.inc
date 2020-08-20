// List of all ops supported by MLIR backend end-to-end
#ifndef MLIR_OP
#define MLIR_OP
#endif

MLIR_OP(ngraph::op::v0::Abs)
MLIR_OP(ngraph::op::v1::Add)
MLIR_OP(ngraph::op::v0::ArgMin)
MLIR_OP(ngraph::op::v0::ArgMax)
MLIR_OP(ngraph::op::v0::AvgPool)
MLIR_OP(ngraph::op::v0::AvgPoolBackprop)
MLIR_OP(ngraph::op::v1::Divide)
MLIR_OP(ngraph::op::v0::Dot)
MLIR_OP(ngraph::op::v0::Concat)
MLIR_OP(ngraph::op::v0::Convolution)
MLIR_OP(ngraph::op::v0::ConvolutionBias)
MLIR_OP(ngraph::op::v0::Gather)
MLIR_OP(ngraph::op::v0::Gemm)
MLIR_OP(ngraph::op::v1::Greater)
MLIR_OP(ngraph::op::v0::GroupConvolution)
MLIR_OP(ngraph::op::v1::Less)
MLIR_OP(ngraph::op::v1::GreaterEqual)
MLIR_OP(ngraph::op::v1::LessEqual)
MLIR_OP(ngraph::op::v1::Equal)
MLIR_OP(ngraph::op::v1::NotEqual)
MLIR_OP(ngraph::op::v0::MatMul)
MLIR_OP(ngraph::op::v1::Maximum)
MLIR_OP(ngraph::op::v0::MaxPool)
MLIR_OP(ngraph::op::v0::MaxPoolBackprop)
MLIR_OP(ngraph::op::v1::Minimum)
MLIR_OP(ngraph::op::v1::Multiply)
MLIR_OP(ngraph::op::v0::Negative)
MLIR_OP(ngraph::op::v0::Softmax)
MLIR_OP(ngraph::op::v1::Subtract)
MLIR_OP(ngraph::op::v0::Relu)

// Add new supported ops here

#undef MLIR_OP
