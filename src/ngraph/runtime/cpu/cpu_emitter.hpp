//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <string>
#include <vector>

#include "ngraph/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_wrapper.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"

#define EMITTER_DECL(op_name)                                                                      \
    emit<op_name>(CPU_ExternalFunction * external_function,                                        \
                  CodeWriter & writer,                                                             \
                  const ngraph::Node* node,                                                        \
                  const std::vector<TensorWrapper>& args,                                          \
                  const std::vector<TensorWrapper>& out)

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_Emitter
            {
            public:
                template <typename OP>
                static void emit(CPU_ExternalFunction* /* external_function */,
                                 CodeWriter& /* writer */,
                                 const ngraph::Node* node,
                                 const std::vector<TensorWrapper>& /* args */,
                                 const std::vector<TensorWrapper>& /* out */)
                {
                    throw std::runtime_error("Unimplemented op '" + node->description() +
                                             "' in CPU emitter");
                }

                static void nop(CPU_ExternalFunction* /* external_function */,
                                CodeWriter& /* writer */,
                                const ngraph::Node* /* node */,
                                const std::vector<TensorWrapper>& /* args */,
                                const std::vector<TensorWrapper>& /* out */)
                {
                }

                template <typename T>
                static void emitBatchNorm(CPU_ExternalFunction* external_function,
                                          CodeWriter& writer,
                                          const ngraph::Node* node,
                                          const std::vector<TensorWrapper>& args,
                                          const std::vector<TensorWrapper>& out,
                                          bool append_relu,
                                          bool training);

            private:
                static std::string emit_vector(const TensorWrapper&, const std::string& name = "");
                static std::string emit_array1d(const TensorWrapper&, const std::string& name = "");
                static std::string emit_matrix(const TensorWrapper&, const std::string& name = "");

                static std::string emit_for_lt(const std::string& prefix, size_t index, size_t to);
                static std::string emit_indices(const std::vector<std::string> indices);
            };

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Add);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::AllReduce);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::BroadcastDistributed);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MatmulBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::BatchMatMul);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Lstm);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Rnn);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::BatchNormTraining);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::BatchNormInference);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormTrainingRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormInferenceRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::BatchNormTrainingBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::CumSum);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Dot);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Multiply);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Abs);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Concat);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Divide);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Equal);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Greater);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::GreaterEqual);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Less);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::LessEqual);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Any);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::All);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::LRN);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Log);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Maximum);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Minimum);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Negative);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::NotEqual);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Select);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Subtract);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Broadcast);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Convert);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Constant);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Reshape);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Sign);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Slice);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Sum);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Exp);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::EmbeddingLookup);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Sin);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Sinh);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Cos);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Cosh);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Tan);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Tanh);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Asin);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Atan);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ArgMin);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ArgMax);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::TopK);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Gather);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::GatherND);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ScatterAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ScatterNDAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::Power);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::UpdateSlice);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ReplaceSlice);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::OneHot);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Ceiling);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Floor);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Sqrt);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::QuantizedConvolutionRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::QuantizedConvolution);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::GroupConvolution);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GroupConvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Convolution);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ConvolutionBackpropFilters);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::DeconvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ConvolutionBackpropData);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::QuantizedConvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::QuantizedConvolutionBiasAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::QuantizedConvolutionBiasSignedAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::QuantizedDotBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::QuantizedDot);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedMatmul);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ConvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ConvolutionBiasAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ConvolutionBiasBackpropFiltersBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::LogicalNot);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolWithIndices);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Reverse);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ReverseSequence);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::AvgPool);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Pad);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::AvgPoolBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::MaxPoolBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolWithIndicesBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Product);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Max);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Erf);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Min);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::runtime::cpu::op::ConvertLayout);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::ReluBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Relu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::CPULeakyRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BoundedRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Sigmoid);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::SigmoidBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidMultiply);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidMultiplyBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Softmax);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Result);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::LogicalAnd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::LogicalOr);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v1::LogicalXor);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::CompiledKernel);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::GenerateMask);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dropout);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Dequantize);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Quantize);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Tile);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::Gelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::v0::RandomUniform);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GeluBackprop);
        }
    }
}
