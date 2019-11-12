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

#include <string>
#include <vector>

#include "ngraph/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"

#define EMITTER_DECL(op_name)                                                                      \
    emit<op_name>(CPU_ExternalFunction * external_function,                                        \
                  CodeWriter & writer,                                                             \
                  const ngraph::Node* node,                                                        \
                  const std::vector<TensorViewWrapper>& args,                                      \
                  const std::vector<TensorViewWrapper>& out)

namespace ngraph
{
    namespace op
    {
        class AllReduce;
        class BroadcastDistributed;
        class MatmulBias;
        class BatchMatMul;
        class BatchMatMulTranspose;
        class Lstm;
        class Rnn;
        class BatchNormTraining;
        class BatchNormInference;
        class BatchNormTrainingRelu;
        class BatchNormInferenceRelu;
        class BatchNormTrainingBackprop;
        class Dot;
        class GetOutputElement;
        class Abs;
        class Concat;
        class Any;
        class All;
        class LRN;
        class Log;
        class Negative;
        class Select;
        class Subtract;
        class Convert;
        class Constant;
        class Reshape;
        class Sign;
        class Exp;
        class EmbeddingLookup;
        class Sin;
        class Sinh;
        class Cos;
        class Cosh;
        class Tan;
        class Tanh;
        class Asin;
        class Atan;
        class ArgMin;
        class ArgMax;
        class GatherND;
        class ScatterAdd;
        class ScatterNDAdd;
        class UpdateSlice;
        class ReplaceSlice;
        class OneHot;
        class Ceiling;
        class Floor;
        class Sqrt;
        class ConvolutionRelu;
        class QuantizedConvolutionRelu;
        class QuantizedConvolution;
        class GroupConvolution;
        class GroupConvolutionBias;
        class DeconvolutionBias;
        class QuantizedConvolutionBias;
        class QuantizedConvolutionBiasAdd;
        class QuantizedConvolutionBiasSignedAdd;
        class QuantizedDotBias;
        class QuantizedDot;
        class QuantizedMatmul;
        class ConvolutionBias;
        class ConvolutionBiasAdd;
        class ConvolutionAdd;
        class ConvolutionBiasBackpropFiltersBias;
        class QuantizedMaxPool;
        class QuantizedAvgPool;
        class MaxPoolWithIndices;
        class ReverseSequence;
        class MaxPoolWithIndicesBackprop;
        class Erf;
        class ReluBackprop;
        class Relu;
        class CPULeakyRelu;
        class BoundedRelu;
        class Sigmoid;
        class SigmoidBackprop;
        class SigmoidMultiply;
        class SigmoidMultiplyBackprop;
        class Result;
        class CompiledKernel;
        class Dropout;
        class Dequantize;
        class Quantize;
        class QuantizedConcat;
        class Tile;
        class Gelu;
        class RandomUniform;
        class GeluBackprop;
    }
    namespace runtime
    {
        namespace cpu
        {
            namespace op
            {
                class ConvertLayout;
            }

            class CPU_Emitter
            {
            public:
                template <typename OP>
                static void emit(CPU_ExternalFunction* /* external_function */,
                                 CodeWriter& /* writer */,
                                 const ngraph::Node* node,
                                 const std::vector<TensorViewWrapper>& /* args */,
                                 const std::vector<TensorViewWrapper>& /* out */)
                {
                    throw std::runtime_error("Unimplemented op '" + node->description() +
                                             "' in CPU emitter");
                }

                static void nop(CPU_ExternalFunction* /* external_function */,
                                CodeWriter& /* writer */,
                                const ngraph::Node* /* node */,
                                const std::vector<TensorViewWrapper>& /* args */,
                                const std::vector<TensorViewWrapper>& /* out */)
                {
                }

                template <typename T>
                static void emitBatchNorm(CPU_ExternalFunction* external_function,
                                          CodeWriter& writer,
                                          const ngraph::Node* node,
                                          const std::vector<TensorViewWrapper>& args,
                                          const std::vector<TensorViewWrapper>& out,
                                          bool append_relu,
                                          bool training);

            private:
                static std::string emit_vector(const TensorViewWrapper&,
                                               const std::string& name = "");
                static std::string emit_array1d(const TensorViewWrapper&,
                                                const std::string& name = "");
                static std::string emit_matrix(const TensorViewWrapper&,
                                               const std::string& name = "");

                static std::string emit_for_lt(const std::string& prefix, size_t index, size_t to);
                static std::string emit_indices(const std::vector<std::string> indices);
            };

            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Add);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AllReduce);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BroadcastDistributed);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MatmulBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchMatMul);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchMatMulTranspose);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Lstm);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Rnn);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormTraining);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormInference);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormTrainingRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormInferenceRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormTrainingBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dot);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Multiply);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GetOutputElement);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Abs);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Concat);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Divide);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Equal);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Greater);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GreaterEq);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Less);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::LessEq);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Any);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::All);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::LRN);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Log);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Maximum);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Minimum);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Negative);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::NotEqual);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Select);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Subtract);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Broadcast);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Convert);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Constant);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reshape);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sign);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Slice);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sum);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Exp);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::EmbeddingLookup);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sin);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sinh);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Cos);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Cosh);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tan);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tanh);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Asin);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Atan);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ArgMin);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ArgMax);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::TopK);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Gather);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GatherND);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ScatterAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ScatterNDAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Power);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::UpdateSlice);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReplaceSlice);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::OneHot);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Ceiling);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Floor);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sqrt);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolution);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GroupConvolution);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GroupConvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Convolution);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropFilters);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::DeconvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropData);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBiasAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConvolutionBiasSignedAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedDotBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedDot);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedMatmul);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBiasAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionAdd);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Not);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedMaxPool);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedAvgPool);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolWithIndices);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Reverse);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReverseSequence);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AvgPool);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Pad);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::AvgPoolBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolWithIndicesBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Product);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Max);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Erf);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Min);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::runtime::cpu::op::ConvertLayout);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::ReluBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Relu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::CPULeakyRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::BoundedRelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Sigmoid);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidMultiply);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::SigmoidMultiplyBackprop);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Softmax);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Result);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::And);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Or);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Xor);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::CompiledKernel);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GenerateMask);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dropout);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Dequantize);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Quantize);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::QuantizedConcat);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Tile);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::Gelu);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::RandomUniform);
            template <>
            void CPU_Emitter::EMITTER_DECL(ngraph::op::GeluBackprop);
        }
    }
}
