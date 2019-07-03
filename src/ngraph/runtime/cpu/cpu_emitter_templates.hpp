//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Add);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Abs);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Acos);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::All);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::AllReduce);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::And);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Any);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ArgMin);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ArgMax);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Asin);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Atan);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::AvgPool);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::AvgPoolBackprop);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BatchMatMul);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BatchMatMulTranspose);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BatchNormInference);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BatchNormInferenceRelu);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BatchNormTraining);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BatchNormTrainingBackprop);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BatchNormTrainingRelu);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BoundedRelu);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Broadcast);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::BroadcastDistributed);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Ceiling);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::CompiledKernel);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Concat);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Constant);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Convert);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Convolution);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ConvolutionAdd);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ConvolutionBackpropFilters);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ConvolutionBackpropData);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ConvolutionBias);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ConvolutionBiasAdd);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ConvolutionBiasBackpropFiltersBias);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ConvolutionRelu);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Cos);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Cosh);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::CPULeakyRelu);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::DeconvolutionBias);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Dequantize);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Divide);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Dot);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Dropout);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::EmbeddingLookup);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Equal);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Erf);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Exp);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Floor);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Gather);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::GatherND);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::GenerateMask);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::GetOutputElement);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Greater);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::GreaterEq);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::GroupConvolution);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::GroupConvolutionBias);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Less);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::LessEq);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Log);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::LRN);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Lstm);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::MatmulBias);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Max);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Maximum);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::MaxPool);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::MaxPoolBackprop);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::MaxPoolWithIndices);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::MaxPoolWithIndicesBackprop);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Min);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Minimum);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Multiply);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Negative);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Not);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::NotEqual);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::OneHot);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Or);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Pad);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Power);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Product);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Quantize);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedAvgPool);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedConcat);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedConvolution);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedConvolutionBias);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedConvolutionBiasAdd);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedConvolutionBiasSignedAdd);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedConvolutionRelu);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedDot);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedDotBias);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedMatmul);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::QuantizedMaxPool);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Relu);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ReluBackprop);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ReplaceSlice);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Reshape);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Result);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Reverse);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ReverseSequence);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Rnn);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ScatterAdd);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::ScatterNDAdd);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Select);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Sigmoid);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::SigmoidBackprop);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::SigmoidMultiply);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::SigmoidMultiplyBackprop);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Sign);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Sin);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Sinh);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Slice);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Softmax);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Sqrt);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Subtract);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Sum);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Tan);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Tanh);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::Tile);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::TopK);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(op::UpdateSlice);
extern template void runtime::cpu::CPU_Emitter::EMITTER_DECL(runtime::cpu::op::ConvertLayout);
