## Supported Ops:

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


| Name | Opset Version(s) supported | Comment 
|------|----------------------------|---------
| Abs | 1-6- |
| Acos | 7- |
| Acosh | (9) | Unimplemented. Not available in nGraph.
| Add | 1-7- |
| And | 1-7- |
| ArgMax | 1- |
| ArgMin | 1- |
| Asin | 7- |
| Asinh | (9) | Unimplemented. Not available in nGraph.
| Atan | 7 - |
| Atanh | (9) | Unimplemented. Not available in nGraph.
| AveragePool | 1-7- |
| BatchNormalization | 1-6-7- | 
| Cast | 1-6- | Errors while casting to bool, Float16 unsupported.
| Ceil | 1-6- | 
| Clip | 1-6- | 
| Compress | (9) | Unimplemented. NGONNX-438. Dynamically computed selected indices.
| Concat | 1-4- | 
| Constant | 1- |  
| ConstantOfShape | (9) | Unimplemented. Dynamic shape input.
| Conv | 1- |
| ConvTranspose | 1- | 
| Cos | 7- |
| Cosh | (9) | Not implemented in Importer C++. Use nGraph `Cosh`.
| DepthToSpace | (1) | Unimplemented. NGONNX-326. WIP
| Div | 1-6-7- | 
| Dropout | 1-6-7- | Only for inference.
| Elu | 1-6- |
| Equal | 1-7 | 
| Erf | (9) | Unimplemented. NGONNX-442. Not available in nGraph - further analysis needed.
| Exp | 1-6- | 
| Expand | - | Unimplemented. Dynamic op.  NGONNX-367, NGRAPH-3289
| EyeLike | (9) | Unimplemented. NGONNX-439 - Make constant node.
| Flatten | 1-(9) | 
| Floor | 1-6- | 
| GRU | - | Unimplemented. NGONNX-325 - Should be possible to implement. Look at `LSTM`
| Gather | - | Unimplemented. Dynamic op. NGONNX-369, NGRAPH-3291
| Gemm | 1-6-7-9 | 
| GlobalAveragePool | 1- | 
| GlobalLpPool | - | Unimplemented. NGONNX-437 - Probably use _slice/op/concat_ pattern.
| GlobalMaxPool | 1- | 
| Greater | 1-7-9 | 
| HardSigmoid | 1-6- | 
| Hardmax | - | Unimplemented. NGONNX-431 - Use make constant and Argmax. See `test_ops_unary.py::test_hardmax()`
| Identity | 1- | 
| If | - | Unimplemented. NGONNX-432 - Probably impossible.
| InstanceNormalization | - | Unimplemented. NGONNX-436. - Just an equation. For per channel computation may _slice/op/concat_ pattern need to be used.
| IsNaN | (9) | Unimplemented. NGONNX-440 - Hacky way is to generate constant nodes with representations of NaN and compare with them.
| LRN | 1- | 
| LSTM | 1-7- | NGONNX-430 Not fully supported.
| LeakyRelu | 1-6- |  
| Less | 1-7-9 |
| Log | 1-6- | 
| LogSoftmax | 1- | 
| Loop | - | Unimplemented. NGONNX-432 - Static loops with some preconditions may be possible, however no idea how to pass graph (proto?) as a _body_ attribute. (what about graph contains `Loop`?)
| LpNormalization | - | Unimplemented. NGONNX-436 - Just an equation. Only Lp{1,2} need to be supported.
| LpPool | - | Unimplemented. NGONNX-437 - Further analysis needed, however probably unsupported by nGraph.
| MatMul | 1-9 | 
| Max | 1-6-8- | 
| MaxPool | 1-8- | 
| MaxRoiPool | - | Unimplemented. NGONNX-437 - Dynamic op. Beside just use _slice/op/concat_ pattern.
| MaxUnpool | (9) | Unimplemented. - not supported by nGraph. 
| Mean | 1-6-8- | 
| Min | 1-6-8- |
| Mul | 1-6-7- | 
| Multinomial | - | Unimplemented. NGONNX-435 - Lack of PRNG in nGraph.
| Neg | 1-6- | 
| Not | 1- | 
| OneHot | (9) | Unimplemented. Furhter analysis needed. Unclear nGraph doc of `OneHot` op.
| Or | 1-7- | 
| PRelu | 1-6-7-9 |
| Pad | 1-2- | Not fully supported. NGCORE-273, NGONNX-416
| Pow | 1-7- | 
| RNN | - | Unimplemented. NGONNX-323, 287 - Should be similar to `LSTM`.
| RandomNormal | - | Unimplemented. NGONNX-434 - Lack of PRNG in nGraph.
| RandomNormalLike | - | Unimplemented. NGONNX-434 - Lack of PRNG in nGraph.
| RandomUniform | - | Unimplemented. NGONNX-434 - Lack of PRNG in nGraph.
| RandomUniformLike | - | Unimplemented. NGONNX-434 - Lack of PRNG in nGraph.
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
| Reshape | 1-5- | NGONNX-357 Lack of support for dynamic shape input. Only as a Constant or as an Initializer.
| Scan | - | Unimplemented. NGONNX-433 Further analysis needed. - determine wheter it is possible to import graph passed by op attribute.
| Scatter | (9) | Unimplemented. Dynamic indices input.
| Selu | 1-6- | 
| Shape | 1- | 
| Shrink | (9) | Unimplemented - No obstacles to implement.
| Sigmoid | 1-6- | 
| Sign | (9) | Unimplemented. Use nGraph `Sign`.
| Sin | 7- | 
| Sinh | (9) | Unimplemented. Use nGraph `Sinh`.
| Size | 1- | 
| Slice | 1- | 
| Softmax | 1- |
| Softplus | 1- |
| Softsign | 1- |
| SpaceToDepth | - | Unimplemented. NGONNX-326
| Split | 1-2- | 
| Sqrt | 1-6- | 
| Squeeze | 1- | 
| Sub | 1-6-7- | 
| Sum | 1-6-8- |
| Tan | 7- |
| Tanh | 1-6- |
| Tile | - | Unimplemented. NGONNX-368, NGRAPH-3292 Dynamic op.
| TopK | - | Unimplemented. NGONNX-327. Use nGraph `Topk`.
| Transpose | 1- | 
| Unsqueeze | 1- |
| Upsample | - | Unimplemented. NGONNX-441 - Dynamic op. 
| Where | (9) | Unimplemented - Use nGraph `Select`.
| Xor | 1-7- |

