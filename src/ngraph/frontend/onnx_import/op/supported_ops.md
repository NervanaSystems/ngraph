Generally nGraph doesn't support tensors of types:

* `string`,
* `float16`,
* `complex64`,
* `complex128`.

Value in `()` _parenthesis_ indicates that this op was introduced since the specific 
ONNX Standard opset version. 
Values seperated by `-` _dash_ indicate the changes were made to that op definition 
in the ONNX Standard. If there were minor changes they are usually supported by single 
implementation, otherwise there are multiple versions, each appropriate for specific opset 
version range.
For example, with the schema represented below the operator `Abs` is supported in all 
opset versions starting from `1` to `6` and to the latest opset version.

## Supported Ops:

| Name | Opset supported | Comment |
|------|----------------------------|---------|
| Abs | 1-6- |
| Acos | 7- |
| Add | 1-7- |
| And | 1-7- |
| ArgMax | 1- |
| ArgMin | 1- |
| Asin | 7- |
| Atan | 7 - |
| AveragePool | 1-7- |
| BatchNormalization | 1-6-7- | 
| Ceil | 1-6- | 
| Clip | 1-6- | 
| Concat | 1-4- | 
| Constant | 1- |  
| Conv | 1- |
| ConvTranspose | 1- | 
| Cos | 7- |
| Div | 1-6-7- | 
| Dropout | 1-6-7- | Only for inference.
| Elu | 1-6- |
| Equal | 1-7 | 
| Exp | 1-6- | 
| Flatten | 1-(9) | 
| Floor | 1-6- | 
| Gemm | 1-6-7-9 | 
| GlobalAveragePool | 1- | 
| GlobalMaxPool | 1- | 
| Greater | 1-7-9 | 
| HardSigmoid | 1-6- | 
| Identity | 1- | 
| LRN | 1- | 
| LeakyRelu | 1-6- |  
| Less | 1-7-9 |
| Log | 1-6- | 
| LogSoftmax | 1- | 
| MatMul | 1-9 | 
| Max | 1-6-8- | 
| MaxPool | 1-8- | 
| Mean | 1-6-8- | 
| Min | 1-6-8- |
| Mul | 1-6-7- | 
| Neg | 1-6- | 
| Not | 1- | 
| Or | 1-7- | 
| PRelu | 1-6-7-9 |
| Pow | 1-7- | 
| Reciprocal | 1-6- | 
| ReduceL1 | 1- | 
| ReduceL2 | 1- | 
| ReduceLogSum | 1- | 
| ReduceLogSumExp | 1- |
| ReduceMax | 1- |
| ReduceMean | 1- |
| ReduceMin | 1- | 
| ReduceProd | 1- |
| ReduceSum | 1- | 
| ReduceSumSquare | 1- |
| Relu | 1-6- | 
| Selu | 1-6- | 
| Shape | 1- | 
| Sigmoid | 1-6- | 
| Sin | 7- | 
| Size | 1- | 
| Slice | 1- | 
| Softmax | 1- |
| Softplus | 1- |
| Softsign | 1- |
| Split | 1-2- | 
| Sqrt | 1-6- | 
| Squeeze | 1- | 
| Sub | 1-6-7- | 
| Sum | 1-6-8- |
| Tan | 7- |
| Tanh | 1-6- |
| Transpose | 1- | 
| Unsqueeze | 1- |
| Xor | 1-7- |

## Unsupported Ops:

### Lack of support in nGraph
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| Acosh | (9) | 283 | 444 | |
| Asinh | (9) | 283 | 444 | |
| Atanh | (9) | 283 | 444 | |
| Erf | (9) | 284 | 442 | Maybe we may implement this as a simple closed interval integral? :) |
| Pad | 1-2- | 273 | 416 | Not fully supported. |
| LSTM | 1-7- | | 430 | Not fully supported. |
| MaxUnpool | (9) | 286, 289 | 447 | |
| LpPool | - | 291 | 437 | Further analysis needed, however probably unsupported by nGraph. |
| Multinomial | - | 199 | 435 | Lack of PRNG in nGraph. |
| RandomNormal | - | 199 | 434 | Lack of PRNG in nGraph. |
| RandomNormalLike | - | 199 | 434 | Lack of PRNG in nGraph. |
| RandomUniform | - | 199 | 434 | Lack of PRNG in nGraph. |
| RandomUniformLike | - | 199 | 434 | Lack of PRNG in nGraph. |
| Cast | 1-6- | 290 | 452 | Float16 unsupported. |

### Futher analysis needed
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| GRU | - | | 325, 177 | Should be possible to implement. Look at `LSTM` |
| RNN | - | | 323, 287 | Should be similar to `LSTM`. |
| If | - | | 432 | At this moment probably impossible. |
| IsNaN | (9) | | 440 | Hacky way is to generate constant nodes with representations of NaN and compare with them. |
| Loop | - | | 432 | Static loops with some preconditions may be possible, however no idea how to pass graph (proto?) as a _body_ attribute. (what about graph contains `Loop`?) |
| OneHot | (9) | | 453 | Furhter analysis needed. Unclear nGraph doc of `OneHot` op. |
| Scan | - |  | 433 | Further analysis needed. - determine whether it is possible to import graph passed by op attribute. |

### Dynamic operators
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| Compress | (9) | 285 | 438. | Dynamically selected indices |
| ConstantOfShape | (9) | 286 | 445 | Dynamic shape input. |
| Expand | - | NGRAPH-3289 | 367 | Dynamic op. |
| Gather | - | NGRAPH-3291 | 369, | Dynamic op.  |
| Tile | - | NGRAPH-3292 | 368 | Dynamic op. |
| Upsample | (7) | 287 | 441 | Dynamic op. |
| MaxRoiPool | - | 288 | 437 | Dynamic op. Beside just use _slice/op/concat_ pattern. |
| Reshape | 1-5- | NGRAPH-3290 | 357 | Lack of support for dynamic shape input. Only as a Constant or as an Initializer. |
| Scatter | (9) | 289 | 446 | Dynamic indices input. |

### Able to implement or WIP
| Name | Opset supported | NGCORE | NGONNX | Comment |
|------|-----------------|--------|--------|---------|
| Cast | 1-6- | | 427 | Errors while casting to bool |
| EyeLike | (9) | | 439 | Make constant node. |
| GlobalLpPool | - | | 437 | Probably use _slice/op/concat_ pattern. |
| Hardmax | - | | 431 | Use make constant and Argmax. See `test_ops_unary.py::test_hardmax()` |
| LpNormalization | - | | 436 | Just an equation. Only Lp{1,2} need to be supported. |
| InstanceNormalization | - | | 436 | Just an equation. For per channel computation may _slice/op/concat_ pattern need to be used. |
| Shrink | (9) | | 449 | Just an easy equation. |
| TopK | - | | 327. | Use nGraph `Topk`. |
| Cosh | (9) | | 448 | Use nGraph `Cosh`. |
| Sign | (9) | | 448 | Use nGraph `Sign`. |
| Sinh | (9) | | 448 | Use nGraph `Sinh`. |
| Where | (9) | | 448 |  Use nGraph `Select`. |
