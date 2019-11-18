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
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/atan2.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast_distributed.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_pad.hpp"
#include "ngraph/op/experimental/dyn_replace_slice.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/layers/ctc_greedy_decoder.hpp"
#include "ngraph/op/experimental/layers/detection_output.hpp"
#include "ngraph/op/experimental/layers/interpolate.hpp"
#include "ngraph/op/experimental/layers/prior_box.hpp"
#include "ngraph/op/experimental/layers/prior_box_clustered.hpp"
#include "ngraph/op/experimental/layers/proposal.hpp"
#include "ngraph/op/experimental/layers/psroi_pooling.hpp"
#include "ngraph/op/experimental/layers/region_yolo.hpp"
#include "ngraph/op/experimental/layers/reorg_yolo.hpp"
#include "ngraph/op/experimental/layers/roi_pooling.hpp"
#include "ngraph/op/experimental/random_uniform.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/experimental/tile.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/fused/clamp.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/depth_to_space.hpp"
#include "ngraph/op/fused/elu.hpp"
#include "ngraph/op/fused/fake_quantize.hpp"
#include "ngraph/op/fused/gelu.hpp"
#include "ngraph/op/fused/gemm.hpp"
#include "ngraph/op/fused/grn.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/fused/group_conv_transpose.hpp"
#include "ngraph/op/fused/gru_cell.hpp"
#include "ngraph/op/fused/hard_sigmoid.hpp"
#include "ngraph/op/fused/layer_norm.hpp"
#include "ngraph/op/fused/log_softmax.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/fused/lstm_sequence.hpp"
#include "ngraph/op/fused/matmul.hpp"
#include "ngraph/op/fused/mvn.hpp"
#include "ngraph/op/fused/normalize_l2.hpp"
#include "ngraph/op/fused/partial_slice.hpp"
#include "ngraph/op/fused/prelu.hpp"
#include "ngraph/op/fused/rnn_cell.hpp"
#include "ngraph/op/fused/scale_shift.hpp"
#include "ngraph/op/fused/selu.hpp"
#include "ngraph/op/fused/shuffle_channels.hpp"
#include "ngraph/op/fused/softmax_crossentropy.hpp"
#include "ngraph/op/fused/space_to_depth.hpp"
#include "ngraph/op/fused/split.hpp"
#include "ngraph/op/fused/squared_difference.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/recv.hpp"
#include "ngraph/op/reduce_mean.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/send.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/op/batch_mat_mul_transpose.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"

#define EMITTER_DECL(op_name)                                                                      \
    emit<op_name>(CPU_ExternalFunction * external_function,                                        \
                  CodeWriter & writer,                                                             \
                  const ngraph::Node* node,                                                        \
                  const std::vector<TensorViewWrapper>& args,                                      \
                  const std::vector<TensorViewWrapper>& out)

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
