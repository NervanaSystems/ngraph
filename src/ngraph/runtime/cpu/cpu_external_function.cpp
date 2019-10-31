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

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#if defined(NGRAPH_TBB_ENABLE)
#define TBB_PREVIEW_FLOW_GRAPH_TRACE 1

#include <tbb/flow_graph.h>
#endif

#if !defined(NGRAPH_DEX_ONLY)
#include "ngraph/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#endif

#ifdef NGRAPH_MLIR_ENABLE
#include "contrib/mlir/core/pass/mlir_subgraph_extraction.hpp"
#endif

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
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
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
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
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/experimental/random_uniform.hpp"
#include "ngraph/op/experimental/tile.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/gelu.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/fused/softmax_crossentropy.hpp"
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
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/batch_fusion.hpp"
#include "ngraph/pass/common_function_collection.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/implicit_broadcast_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/propagate_cacheability.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/reshape_sinking.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_builder_registry.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_cse.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"
#include "ngraph/runtime/cpu/cpu_visualize_tree.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_mat_mul_transpose.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"
#include "ngraph/runtime/cpu/pass/cpu_collapse_dims.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_horizontal_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_layout.hpp"
#include "ngraph/runtime/cpu/pass/cpu_mat_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_memory_assignment.hpp"
#include "ngraph/runtime/cpu/pass/cpu_memory_optimization.hpp"
#include "ngraph/runtime/cpu/pass/cpu_mkldnn_primitive_build.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"
#include "ngraph/runtime/cpu/pass/cpu_rnn_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_workspace_insertion.hpp"
#include "ngraph/runtime/cpu/pass/halide_subgraph_extraction.hpp"

using namespace std;
using namespace ngraph;

#define STR(s) #s

#define REGISTER_KNOBBED_PASS(name, enable_by_default, prefix)                                     \
    if (pass_map.find(STR(name)) != pass_map.end())                                                \
    {                                                                                              \
        if (pass_map[STR(name)])                                                                   \
        {                                                                                          \
            pass_manager.register_pass<prefix::name>();                                            \
        }                                                                                          \
    }                                                                                              \
    else if (enable_by_default)                                                                    \
    {                                                                                              \
        pass_manager.register_pass<prefix::name>();                                                \
    }

#define REGISTER_KNOBBED_PASS_WITH_ARGS(name, enable_by_default, prefix, ...)                      \
    if (pass_map.find(STR(name)) != pass_map.end())                                                \
    {                                                                                              \
        if (pass_map[STR(name)])                                                                   \
        {                                                                                          \
            pass_manager.register_pass<prefix::name>(__VA_ARGS__);                                 \
        }                                                                                          \
    }                                                                                              \
    else if (enable_by_default)                                                                    \
    {                                                                                              \
        pass_manager.register_pass<prefix::name>(__VA_ARGS__);                                     \
    }

runtime::cpu::CPU_ExternalFunction::CPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function, bool release_function)
    : m_function(function)
    , m_release_function(release_function)
    , m_emit_timing(false)
#if defined(NGRAPH_TBB_ENABLE)
    , m_use_tbb(std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
#endif
#if !defined(NGRAPH_DEX_ONLY)
    , m_is_compiled(false)
    , m_direct_execution((std::getenv("NGRAPH_CODEGEN") == nullptr) ||
                         (std::string(std::getenv("NGRAPH_CODEGEN")) == "0"))
#else
    , m_direct_execution(true)
#endif
    , m_compiled_function(nullptr)
    , m_function_name(function->get_name())
    , m_is_built(false)
{
}

runtime::cpu::CPU_ExternalFunction::~CPU_ExternalFunction()
{
    for (auto state : m_states)
    {
        delete state;
    }
}

class StaticInitializers
{
public:
    StaticInitializers(string directory) { ngraph::file_util::remove_directory(directory); }
};

#if !defined(NGRAPH_DEX_ONLY)

static const string s_output_dir = "cpu_codegen";

static string emit_string_array(const vector<string>& s, size_t max_line_length)
{
    stringstream ss;
    stringstream line;
    for (size_t i = 0; i < s.size(); i++)
    {
        if (i != 0)
        {
            line << ",";
        }
        stringstream value;
        value << s[i];
        string value_string = value.str();
        if (static_cast<size_t>(line.tellp()) + value_string.size() + 1 <= max_line_length)
        {
            if (i > 0)
            {
                line << " ";
            }
            line << value_string;
        }
        else
        {
            ss << line.str() << "\n";
            line.str("");
            line << value_string;
        }
    }
    ss << line.str();
    return ss.str();
}

static StaticInitializers s_static_initializers(s_output_dir);

#define TI(x) type_index(typeid(x))

static const runtime::cpu::OpMap dispatcher{
    {TI(ngraph::op::Add), &runtime::cpu::CPU_Emitter::emit<op::Add>},
    {TI(ngraph::op::AllReduce), &runtime::cpu::CPU_Emitter::emit<op::AllReduce>},
    {TI(ngraph::op::BroadcastDistributed),
     &runtime::cpu::CPU_Emitter::emit<op::BroadcastDistributed>},
    {TI(ngraph::op::MatmulBias), &runtime::cpu::CPU_Emitter::emit<op::MatmulBias>},
    {TI(ngraph::op::Dot), &runtime::cpu::CPU_Emitter::emit<op::Dot>},
    {TI(ngraph::op::Multiply), &runtime::cpu::CPU_Emitter::emit<op::Multiply>},
    {TI(ngraph::op::Parameter), &runtime::cpu::CPU_Emitter::nop},
    {TI(ngraph::op::Abs), &runtime::cpu::CPU_Emitter::emit<op::Abs>},
    {TI(ngraph::op::Any), &runtime::cpu::CPU_Emitter::emit<op::Any>},
    {TI(ngraph::op::All), &runtime::cpu::CPU_Emitter::emit<op::All>},
    {TI(ngraph::op::BatchMatMul), &runtime::cpu::CPU_Emitter::emit<op::BatchMatMul>},
    {TI(ngraph::op::BatchMatMulTranspose),
     &runtime::cpu::CPU_Emitter::emit<op::BatchMatMulTranspose>},
    {TI(ngraph::op::Concat), &runtime::cpu::CPU_Emitter::emit<op::Concat>},
    {TI(ngraph::op::Divide), &runtime::cpu::CPU_Emitter::emit<op::Divide>},
    {TI(ngraph::op::Equal), &runtime::cpu::CPU_Emitter::emit<op::Equal>},
    {TI(ngraph::op::Erf), &runtime::cpu::CPU_Emitter::emit<op::Erf>},
    {TI(ngraph::op::Gather), &runtime::cpu::CPU_Emitter::emit<op::Gather>},
    {TI(ngraph::op::GatherND), &runtime::cpu::CPU_Emitter::emit<op::GatherND>},
    {TI(ngraph::op::ScatterAdd), &runtime::cpu::CPU_Emitter::emit<op::ScatterAdd>},
    {TI(ngraph::op::ScatterNDAdd), &runtime::cpu::CPU_Emitter::emit<op::ScatterNDAdd>},
    {TI(ngraph::op::GetOutputElement), &runtime::cpu::CPU_Emitter::emit<op::GetOutputElement>},
    {TI(ngraph::op::Greater), &runtime::cpu::CPU_Emitter::emit<op::Greater>},
    {TI(ngraph::op::GreaterEq), &runtime::cpu::CPU_Emitter::emit<op::GreaterEq>},
    {TI(ngraph::op::Less), &runtime::cpu::CPU_Emitter::emit<op::Less>},
    {TI(ngraph::op::LessEq), &runtime::cpu::CPU_Emitter::emit<op::LessEq>},
    {TI(ngraph::op::Log), &runtime::cpu::CPU_Emitter::emit<op::Log>},
    {TI(ngraph::op::Maximum), &runtime::cpu::CPU_Emitter::emit<op::Maximum>},
    {TI(ngraph::op::Minimum), &runtime::cpu::CPU_Emitter::emit<op::Minimum>},
    {TI(ngraph::op::Negative), &runtime::cpu::CPU_Emitter::emit<op::Negative>},
    {TI(ngraph::op::NotEqual), &runtime::cpu::CPU_Emitter::emit<op::NotEqual>},
    {TI(ngraph::op::Power), &runtime::cpu::CPU_Emitter::emit<op::Power>},
    {TI(ngraph::op::Select), &runtime::cpu::CPU_Emitter::emit<op::Select>},
    {TI(ngraph::op::Subtract), &runtime::cpu::CPU_Emitter::emit<op::Subtract>},
    {TI(ngraph::op::Broadcast), &runtime::cpu::CPU_Emitter::emit<op::Broadcast>},
    {TI(ngraph::op::Convert), &runtime::cpu::CPU_Emitter::emit<op::Convert>},
    {TI(ngraph::op::Constant), &runtime::cpu::CPU_Emitter::emit<op::Constant>},
    {TI(ngraph::op::Reshape), &runtime::cpu::CPU_Emitter::emit<op::Reshape>},
    {TI(ngraph::op::Sign), &runtime::cpu::CPU_Emitter::emit<op::Sign>},
    {TI(ngraph::op::Slice), &runtime::cpu::CPU_Emitter::emit<op::Slice>},
    {TI(ngraph::op::Sum), &runtime::cpu::CPU_Emitter::emit<op::Sum>},
    {TI(ngraph::op::EmbeddingLookup), &runtime::cpu::CPU_Emitter::emit<op::EmbeddingLookup>},
    {TI(ngraph::op::Exp), &runtime::cpu::CPU_Emitter::emit<op::Exp>},
    {TI(ngraph::op::Sin), &runtime::cpu::CPU_Emitter::emit<op::Sin>},
    {TI(ngraph::op::Sinh), &runtime::cpu::CPU_Emitter::emit<op::Sinh>},
    {TI(ngraph::op::Cos), &runtime::cpu::CPU_Emitter::emit<op::Cos>},
    {TI(ngraph::op::Cosh), &runtime::cpu::CPU_Emitter::emit<op::Cosh>},
    {TI(ngraph::op::Tan), &runtime::cpu::CPU_Emitter::emit<op::Tan>},
    {TI(ngraph::op::Tanh), &runtime::cpu::CPU_Emitter::emit<op::Tanh>},
    {TI(ngraph::op::TopK), &runtime::cpu::CPU_Emitter::emit<op::TopK>},
    {TI(ngraph::op::Asin), &runtime::cpu::CPU_Emitter::emit<op::Asin>},
    {TI(ngraph::op::ArgMin), &runtime::cpu::CPU_Emitter::emit<op::ArgMin>},
    {TI(ngraph::op::ArgMax), &runtime::cpu::CPU_Emitter::emit<op::ArgMax>},
    {TI(ngraph::op::Acos), &runtime::cpu::CPU_Emitter::emit<op::Acos>},
    {TI(ngraph::op::Atan), &runtime::cpu::CPU_Emitter::emit<op::Atan>},
    {TI(ngraph::op::ReplaceSlice), &runtime::cpu::CPU_Emitter::emit<op::ReplaceSlice>},
    {TI(ngraph::op::UpdateSlice), &runtime::cpu::CPU_Emitter::emit<op::UpdateSlice>},
    {TI(ngraph::op::OneHot), &runtime::cpu::CPU_Emitter::emit<op::OneHot>},
    {TI(ngraph::op::Floor), &runtime::cpu::CPU_Emitter::emit<op::Floor>},
    {TI(ngraph::op::Ceiling), &runtime::cpu::CPU_Emitter::emit<op::Ceiling>},
    {TI(ngraph::op::Sqrt), &runtime::cpu::CPU_Emitter::emit<op::Sqrt>},
    {TI(ngraph::op::Convolution), &runtime::cpu::CPU_Emitter::emit<op::Convolution>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::cpu::CPU_Emitter::emit<op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::cpu::CPU_Emitter::emit<op::ConvolutionBackpropData>},
    {TI(ngraph::op::GroupConvolution), &runtime::cpu::CPU_Emitter::emit<op::GroupConvolution>},
    {TI(ngraph::op::ConvolutionBias), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionBias>},
    {TI(ngraph::op::QuantizedConvolutionBias),
     &runtime::cpu::CPU_Emitter::emit<op::QuantizedConvolutionBias>},
    {TI(ngraph::op::QuantizedConvolutionBiasAdd),
     &runtime::cpu::CPU_Emitter::emit<op::QuantizedConvolutionBiasAdd>},
    {TI(ngraph::op::QuantizedConvolutionBiasSignedAdd),
     &runtime::cpu::CPU_Emitter::emit<op::QuantizedConvolutionBiasSignedAdd>},
    {TI(ngraph::op::QuantizedDotBias), &runtime::cpu::CPU_Emitter::emit<op::QuantizedDotBias>},
    {TI(ngraph::op::QuantizedDot), &runtime::cpu::CPU_Emitter::emit<op::QuantizedDot>},
    {TI(ngraph::op::QuantizedMatmul), &runtime::cpu::CPU_Emitter::emit<op::QuantizedMatmul>},
    {TI(ngraph::op::ConvolutionRelu), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionRelu>},
    {TI(ngraph::op::QuantizedConvolution),
     &runtime::cpu::CPU_Emitter::emit<op::QuantizedConvolution>},
    {TI(ngraph::op::QuantizedConvolutionRelu),
     &runtime::cpu::CPU_Emitter::emit<op::QuantizedConvolutionRelu>},
    {TI(ngraph::op::ConvolutionBiasAdd), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionBiasAdd>},
    // conv+bias backprop for data share the same implementation as ConvolutionBackpropData
    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
     &runtime::cpu::CPU_Emitter::emit<op::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::runtime::cpu::op::ConvertLayout),
     &runtime::cpu::CPU_Emitter::emit<runtime::cpu::op::ConvertLayout>},
    {TI(ngraph::op::Not), &runtime::cpu::CPU_Emitter::emit<op::Not>},
    {TI(ngraph::op::MaxPool), &runtime::cpu::CPU_Emitter::emit<op::MaxPool>},
    {TI(ngraph::op::MaxPoolWithIndices), &runtime::cpu::CPU_Emitter::emit<op::MaxPoolWithIndices>},
    {TI(ngraph::op::Reverse), &runtime::cpu::CPU_Emitter::emit<op::Reverse>},
    {TI(ngraph::op::ReverseSequence), &runtime::cpu::CPU_Emitter::emit<op::ReverseSequence>},
    {TI(ngraph::op::Result), &runtime::cpu::CPU_Emitter::emit<op::Result>},
    {TI(ngraph::op::AvgPool), &runtime::cpu::CPU_Emitter::emit<op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop), &runtime::cpu::CPU_Emitter::emit<op::AvgPoolBackprop>},
    {TI(ngraph::op::Pad), &runtime::cpu::CPU_Emitter::emit<op::Pad>},
    {TI(ngraph::op::BatchNormTraining), &runtime::cpu::CPU_Emitter::emit<op::BatchNormTraining>},
    {TI(ngraph::op::BatchNormInference), &runtime::cpu::CPU_Emitter::emit<op::BatchNormInference>},
    {TI(ngraph::op::BatchNormTrainingRelu),
     &runtime::cpu::CPU_Emitter::emit<op::BatchNormTrainingRelu>},
    {TI(ngraph::op::BatchNormInferenceRelu),
     &runtime::cpu::CPU_Emitter::emit<op::BatchNormInferenceRelu>},
    {TI(ngraph::op::BatchNormTrainingBackprop),
     &runtime::cpu::CPU_Emitter::emit<op::BatchNormTrainingBackprop>},
    {TI(ngraph::op::BoundedRelu), &runtime::cpu::CPU_Emitter::emit<op::BoundedRelu>},
    {TI(ngraph::op::Lstm), &runtime::cpu::CPU_Emitter::emit<op::Lstm>},
    {TI(ngraph::op::MaxPoolBackprop), &runtime::cpu::CPU_Emitter::emit<op::MaxPoolBackprop>},
    {TI(ngraph::op::MaxPoolWithIndicesBackprop),
     &runtime::cpu::CPU_Emitter::emit<op::MaxPoolWithIndicesBackprop>},
    {TI(ngraph::op::Product), &runtime::cpu::CPU_Emitter::emit<op::Product>},
    {TI(ngraph::op::Max), &runtime::cpu::CPU_Emitter::emit<op::Max>},
    {TI(ngraph::op::Min), &runtime::cpu::CPU_Emitter::emit<op::Min>},
    {TI(ngraph::op::Relu), &runtime::cpu::CPU_Emitter::emit<op::Relu>},
    {TI(ngraph::op::ReluBackprop), &runtime::cpu::CPU_Emitter::emit<op::ReluBackprop>},
    {TI(ngraph::op::Rnn), &runtime::cpu::CPU_Emitter::emit<op::Rnn>},
    {TI(ngraph::op::Sigmoid), &runtime::cpu::CPU_Emitter::emit<op::Sigmoid>},
    {TI(ngraph::op::SigmoidMultiply), &runtime::cpu::CPU_Emitter::emit<op::SigmoidMultiply>},
    {TI(ngraph::op::SigmoidMultiplyBackprop),
     &runtime::cpu::CPU_Emitter::emit<op::SigmoidMultiplyBackprop>},
    {TI(ngraph::op::Softmax), &runtime::cpu::CPU_Emitter::emit<op::Softmax>},
    {TI(ngraph::op::SigmoidBackprop), &runtime::cpu::CPU_Emitter::emit<op::SigmoidBackprop>},
    {TI(ngraph::op::And), &runtime::cpu::CPU_Emitter::emit<op::And>},
    {TI(ngraph::op::Or), &runtime::cpu::CPU_Emitter::emit<op::Or>},
    {TI(ngraph::op::Xor), &runtime::cpu::CPU_Emitter::emit<op::Xor>},
    {TI(ngraph::op::CPULeakyRelu), &runtime::cpu::CPU_Emitter::emit<op::CPULeakyRelu>},
    {TI(ngraph::op::CompiledKernel), &runtime::cpu::CPU_Emitter::emit<op::CompiledKernel>},
    {TI(ngraph::op::LRN), &runtime::cpu::CPU_Emitter::emit<ngraph::op::LRN>},
    {TI(ngraph::op::GenerateMask), &runtime::cpu::CPU_Emitter::emit<ngraph::op::GenerateMask>},
    {TI(ngraph::op::ConvolutionAdd), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionAdd>},
    {TI(ngraph::op::Quantize), &runtime::cpu::CPU_Emitter::emit<ngraph::op::Quantize>},
    {TI(ngraph::op::Dequantize), &runtime::cpu::CPU_Emitter::emit<ngraph::op::Dequantize>},
    {TI(ngraph::op::GroupConvolutionBias),
     &runtime::cpu::CPU_Emitter::emit<op::GroupConvolutionBias>},
    {TI(ngraph::op::DeconvolutionBias),
     &runtime::cpu::CPU_Emitter::emit<ngraph::op::DeconvolutionBias>},
    {TI(ngraph::op::Dropout), &runtime::cpu::CPU_Emitter::emit<op::Dropout>},
    {TI(ngraph::op::Tile), &runtime::cpu::CPU_Emitter::emit<op::Tile>},
    {TI(ngraph::op::Gelu), &runtime::cpu::CPU_Emitter::emit<op::Gelu>},
    {TI(ngraph::op::GeluBackprop), &runtime::cpu::CPU_Emitter::emit<op::GeluBackprop>},
};

static void
    generate_isnan_isinf_check(CodeWriter& writer,
                               std::shared_ptr<Node> node,
                               const std::vector<ngraph::runtime::cpu::TensorViewWrapper>& out,
                               const char* funcname)
{
    auto ctype = node->get_element_type().c_type_string();
    writer << "{   // A " << funcname << " for" << node->get_name() << "\n";
    writer.indent++;
    writer << " ngraph::check_fp_values<" << ctype << "," << funcname << "> (\"" << node->get_name()
           << "\", (" << ctype << "*)" << out[0].get_name() << ", " << out[0].get_size() << ");\n";
    writer.indent--;
    writer << "}\n";
}

static void generate_class_declarations(CodeWriter& writer)
{
    writer << "// Declare all classes\n";
    writer << "struct CPURuntimeContextCG;\n";
}

static void generate_runtime_context_class(CodeWriter& writer)
{
    writer <<
#include "ngraph/runtime/cpu/pregenerated_src/cpu_cg_runtime_context.hpp"
           << "\n";
}

void runtime::cpu::CPU_ExternalFunction::compile(ngraph::pass::PassConfig& pass_config)
{
    if (m_is_compiled)
    {
        return;
    }

    m_mkldnn_emitter.reset(new MKLDNNEmitter());

    ngraph::pass::Manager pass_manager;
    register_common_passes(pass_manager, pass_config);

    // Build mkldnn primitives for codegen.
    pass_manager.register_pass<runtime::cpu::pass::MKLDNNPrimitiveBuildPass>(
        m_desc_filename, *m_mkldnn_emitter, m_node_primitive_string_deps_index_size_map);

    unordered_map<Node*, Node*> node_function_map;
    string common_function_string;
    auto femitter = bind(&ngraph::runtime::cpu::CPU_ExternalFunction::emit_op_as_function,
                         this,
                         placeholders::_1,
                         placeholders::_2);
    pass_manager.register_pass<ngraph::pass::CommonFunctionCollection>(
        femitter, node_function_map, common_function_string);
    pass_manager.run_passes(m_function);

    list<shared_ptr<Node>> ordered_ops = m_function->get_ordered_ops();

    CodeWriter writer;

    writer << "// Generated by the nGraph CPU backend\n";
#if defined(NGRAPH_TBB_ENABLE)
    if (m_use_tbb)
    {
        if (runtime::cpu::IsTracingEnabled() || m_emit_timing)
        {
            throw ngraph_error(
                "CPU Backend: Tracing and performance breakdowns might not be accurate with TBB "
                "enabled due to concurrent graph execution");
        }
        writer << "#include <tbb/flow_graph.h>";
    }
#endif

    writer +=
        R"(
#include <cmath>
#include <fstream>
#include <mkldnn.hpp>
#include "ngraph/distributed.hpp"
#include "ngraph/except.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_eigen_utils.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/and.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/reference/argmax.hpp"
#include "ngraph/runtime/reference/argmin.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/embedding_lookup.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "ngraph/runtime/reference/generate_mask.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/scatter_add.hpp"
#include "ngraph/runtime/reference/scatter_nd_add.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/topk.hpp"
#include "ngraph/runtime/reference/xor.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/state/bernoulli_rng_state.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/util.hpp"

using namespace ngraph::runtime::cpu::eigen;
using namespace ngraph::runtime;

)";

    string pch_header_source = writer.get_code();

    // The "dso_handle" symbol is required by __cxa_atexit()
    // which is enabled because the JIT uses it as the default mechanism
    // to register cleanup handlers. We use it, and not atexit(), because
    // atexit() happens too late, when the JIT is no longer alive

    writer << "void *__dso_handle = 0;\n\n";

    if (m_emit_timing)
    {
        writer << "// Declare debug timers\n";
        vector<string> names;
        size_t index = 0;
        for (shared_ptr<Node> node : ordered_ops)
        {
            if (!node->is_parameter() && !node->is_constant())
            {
                names.push_back(node->get_name());
                m_name_index_map.insert({node->get_name(), index++});
            }
        }
        writer << "ngraph::stopwatch timers[" << names.size() << "];\n";
        writer << "extern \"C\" size_t get_debug_timer_count() { return " << names.size()
               << "; }\n";
        writer << "extern \"C\" const char* get_debug_timer_name(size_t index)\n";
        writer << "{\n";
        writer.indent++;
        writer << "static const char* timer_names[" << names.size() << "] =\n";
        writer << "{\n";
        writer.indent++;
        vector<string> quoted_names;
        for (const string& name : names)
        {
            quoted_names.push_back("\"" + name + "\"");
        }
        writer << emit_string_array(quoted_names, 100 - (4 * 2 + 1));
        writer << "\n};\n";
        writer.indent--;
        writer << "return timer_names[index];\n";
        writer.indent--;
        writer << "}\n";

        writer << "extern \"C\" const size_t get_debug_timer_microseconds(size_t index)\n";
        writer << "{\n";
        writer.indent++;
        writer << "return (index < " << names.size()
               << " ? timers[index].get_total_microseconds() : 0);\n";
        writer.indent--;
        writer << "}\n";

        writer << "extern \"C\" const size_t get_debug_timer_call_count(size_t index)\n";
        writer << "{\n";
        writer.indent++;
        writer << "return (index < " << names.size() << " ? timers[index].get_call_count() : 0);\n";
        writer.indent--;
        writer << "}\n";
        writer << "\n";
    }

    writer << "// Declare all constants\n";
    for (shared_ptr<Node> node : ordered_ops)
    {
        ngraph::op::Constant* c = as_type<ngraph::op::Constant>(node.get());
        if (c)
        {
            m_active_constants.push_back(node);
            shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
            string type = tv->get_element_type().c_type_string();
            writer << "static " << type << "* " << tv->get_name() << " = ((" << type << "*)("
                   << c->get_data_ptr() << "));\n";

            auto output_tensor = &node->get_output_tensor();
            auto tensor_set = get_tensor_set(output_tensor);
            // process all tensors in the set containing the output tensor of the constant
            for (auto& ele_t : tensor_set)
            {
                NGRAPH_CHECK(ele_t->get_pool_offset() == 0, "no offset set for constants");
                m_tensor_roles[ele_t->get_name()] = TensorRole::CONSTANT;
                m_variable_name_map[ele_t->get_name()] = output_tensor->get_name();
            }
        }
    }

    generate_class_declarations(writer);

    const char* func_params =
        "(void** inputs, void** outputs, cpu::CPURuntimeContext* ctx, CPURuntimeContextCG* cg_ctx)";

    writer << "// Declare all functions\n";
    for (shared_ptr<Function> f : pass_manager.get_state().get_functions())
    {
        writer << "extern \"C\" void " << f->get_name() << func_params << ";\n";
    }
    writer << "\n";

    generate_runtime_context_class(writer);

    writer << common_function_string << "\n";

    // initiate mkldnn_primitives for CPURuntimeContextCG
    writer << "void inline CPURuntimeContextCG::init_mkldnn_primitives()\n";
    writer.block_begin();
    writer << "mkldnn_primitives = std::vector<mkldnn::primitive*>("
           << to_string(m_mkldnn_emitter->get_mkldnn_primitives().size()) << ");\n";
    writer << "mkldnn_memories = std::vector<mkldnn::memory*>("
           << to_string(m_mkldnn_emitter->get_mkldnn_memories().size()) << ");\n";
    writer << "mkldnn_scratchpad_mds = std::vector<mkldnn::memory::desc*>("
           << to_string(m_mkldnn_emitter->get_mkldnn_scratchpad_mds().size()) << ");\n";
    writer << "size_t scratchpad_size = " << m_mkldnn_emitter->get_max_scratchpad_size() << ";\n";
    writer << "if (scratchpad_size > 0)\n";
    writer.block_begin();
    writer << "size_t alignment = 4096;\n";
    writer << "scratchpad_buffer = new AlignedBuffer(scratchpad_size, alignment);\n";
    writer.block_end();
    writer << "else\n";
    writer.block_begin();
    writer << "scratchpad_buffer = nullptr;\n";
    writer.block_end();
    writer.block_end();
    writer << "\n";

    set<string> output_names;
    for (shared_ptr<Node> op : m_function->get_results())
    {
        shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
        output_names.insert(tv->get_name());
    }
    set<descriptor::Tensor*> constants;
    for (shared_ptr<Node> node : ordered_ops)
    {
        if (is_type<ngraph::op::Constant>(node))
        {
            shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
            constants.insert(tv.get());
        }
    }

    bool temporaries_used = false;
    for (shared_ptr<Node> node : ordered_ops)
    {
        if (node->liveness_new_list.size() > 0)
        {
            temporaries_used = true;
        }
    }
    if (temporaries_used)
    {
        m_memory_buffer_sizes.push_back(m_function->get_temporary_pool_size());
    }

    // Indexing for Control Flags
    std::map<std::string, size_t> tensor_index_map;
    std::map<std::string, size_t> param_index_map;
    size_t tensor_index = 0;
    for (shared_ptr<Node> node : ordered_ops)
    {
        if (!node->is_parameter() && !node->is_constant())
        {
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                tensor_index_map.insert({tv->get_name(), tensor_index++});
            }
        }
    }

    writer << "bool " << m_function->get_name() << "_t_en[" << tensor_index << "];\n";

    writer << "extern \"C\" void " << m_function->get_name() << func_params << "\n";
    writer << "{\n";
    writer.indent++;

    // deserialize and build mkldnn primitives
    if (m_mkldnn_emitter->get_mkldnn_descriptors_size() > 0)
    {
        writer << "if (ctx->first_iteration)\n";
        writer.block_begin();
        writer << "// read in memory descriptors and build mkldnn primitives\n";
        writer << "std::ifstream desc_file (\"" << m_desc_filename << "\", std::ios::binary);\n";
        writer << "deserialize_memory_descs_and_build_memory(" << m_desc_filename << ", cg_ctx, "
               << to_string(m_mkldnn_emitter->get_mkldnn_descriptors_size()) << ");\n";
        writer.block_end();
    }

    // Execution tracing support
    if (runtime::cpu::IsTracingEnabled() && m_function->get_name() == m_function_name)
    {
        writer << "cpu::Timestamp start_ts;\n"
               << "int profiler_count = 0;\n\n";
    }

    if (temporaries_used)
    {
        writer << "size_t pool_base_ptr = (size_t) ctx->memory_buffers["
               << m_memory_buffer_sizes.size() - 1 << "]->get_ptr();\n";
        writer << "\n";
    }

    writer << "bool* t_en = (bool*)" << m_function->get_name() << "_t_en;\n";

#if defined(NGRAPH_TBB_ENABLE)
    if (m_use_tbb)
    {
        writer << "\n";
        writer << "if (ctx->first_iteration) {\n";
        writer.indent++;
        writer << "tbb::flow::continue_node<tbb::flow::continue_msg>* "
                  "flowgraph_node_start"
               << " = new tbb::flow::continue_node<tbb::flow::continue_msg> "
                  "(*(cg_ctx->tbb_graph), [&](const tbb::flow::continue_msg &msg)\n{});\n";
    }
#endif

    // Add inputs to the variable name map
    size_t arg_index = 0;
    for (shared_ptr<ngraph::op::Parameter> param : m_function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            auto output_tensor = &param->get_outputs().at(i).get_tensor();
            param_index_map[output_tensor->get_name()] = arg_index;
            auto tensor_set = get_tensor_set(output_tensor);

            // process all tensors in the set containing the output tensor of the parameter
            for (auto& ele_t : tensor_set)
            {
                const element::Type& et = ele_t->get_element_type();
                string type = et.c_type_string();
                stringstream ss;
                ss << "(((" << type << "*)(inputs[" << arg_index << "])) + "
                   << ele_t->get_pool_offset() / et.size() << ")";
                m_variable_name_map[ele_t->get_name()] = ss.str();
                m_tensor_roles[ele_t->get_name()] = TensorRole::INPUT;
            }
            arg_index++;
        }
    }

    // Add temporaries to the variable name map
    if (temporaries_used)
    {
        for (auto& ele : bufferID_to_tensorSets)
        {
            if (ele.second.first == TensorRole::INTERMEDIATE)
            {
                for (auto& ele_t : ele.second.second)
                {
                    stringstream ss;
                    ss << "((" << ele_t->get_element_type().c_type_string() << "*)(pool_base_ptr + "
                       << ele_t->get_pool_offset() << "))";
                    m_variable_name_map[ele_t->get_name()] = ss.str();
                    m_tensor_roles[ele_t->get_name()] = TensorRole::INTERMEDIATE;
                }
            }
        }
    }

    // Add outputs to the variable name map
    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        shared_ptr<Node> op = m_function->get_output_op(i);
        auto output_tensor = &op->get_output_tensor();
        auto tensor_set = get_tensor_set(output_tensor);
        // process all tensors in the set containing the output tensor of the result
        for (auto& ele_t : tensor_set)
        {
            const element::Type& et = ele_t->get_element_type();
            string type = et.c_type_string();
            stringstream ss;
            ss << "(((" << type << "*)(outputs[" << i << "])) + "
               << ele_t->get_pool_offset() / et.size() << ")";
            m_variable_name_map[ele_t->get_name()] = ss.str();
            m_tensor_roles[ele_t->get_name()] = TensorRole::OUTPUT;
        }
    }

    for (shared_ptr<Node> node : ordered_ops)
    {
        auto& n = *node; // Work around a compiler warning (*node inside typeid may have effects
        // with shared pointers, which is fine here but clang doesn't like it.)
        auto handler = dispatcher.find(type_index(typeid(n)));
        if (handler == dispatcher.end())
        {
            throw unsupported_op(node->description());
        }
        vector<TensorViewWrapper> in;
        vector<string> node_input_names;
        vector<string> node_output_names;
        vector<TensorTracerAttributes> t_in_attrs;
        vector<TensorTracerAttributes> t_out_attrs;
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            in.push_back(TensorViewWrapper(tv, m_variable_name_map[tv->get_name()]));
            node_input_names.emplace_back(tv->get_name());
            t_in_attrs.push_back(TensorTracerAttributes(
                in.back().get_size(), in.back().get_shape(), in.back().get_element_type()));
        }
        vector<TensorViewWrapper> out;
        for (const descriptor::Output& output : node->get_outputs())
        {
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            out.push_back(TensorViewWrapper(tv, m_variable_name_map[tv->get_name()]));
            node_output_names.emplace_back(tv->get_name());
            t_out_attrs.push_back(TensorTracerAttributes(
                out.back().get_size(), out.back().get_shape(), out.back().get_element_type()));
        }

        // Emit operation prologue
        if (!node->is_parameter() && !node->is_constant())
        {
            if (m_function->get_name() == m_function_name)
            {
                m_op_attrs.emplace_back(node->description(),
                                        node_output_names,
                                        node_input_names,
                                        t_out_attrs,
                                        t_in_attrs);
            }
            if (m_use_tbb)
            {
                writer << "tbb::flow::continue_node<tbb::flow::continue_msg>* "
                          "flowgraph_node_"
                       << node->get_name()
                       << " = new tbb::flow::continue_node<tbb::flow::continue_msg> "
                          "(*(cg_ctx->tbb_graph), [&](const tbb::flow::continue_msg &msg)\n{\n";
                writer.indent++;
            }
            if (runtime::cpu::IsTracingEnabled() && m_function->get_name() == m_function_name)
            {
                writer << "start_ts = cpu::Clock::now();\n";
            }
        }

        if (!node->is_parameter() && !node->is_constant())
        {
            writer << "\n// " << node->get_name() << "(";
            vector<string> parameter_nodes = node_input_names;
            parameter_nodes.insert(
                parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
            writer << join(parameter_nodes);
            writer << ")\n";
        }

        // Emit operation body
        if (!node->is_parameter() && !node->is_constant())
        {
            emit_debug_function_entry(writer, node.get(), in, out);
        }

        // Op Control
        if (!node->is_parameter() && !node->is_constant())
        {
            writer << "if (ctx->first_iteration ";
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                auto input_name = tv->get_name();

                if (output.get_node()->is_parameter())
                {
                    writer << " || ctx->p_en[" << param_index_map[input_name] << "]";
                }
                else if (!output.get_node()->is_constant())
                {
                    writer << " || t_en[" << tensor_index_map[input_name] << "]";
                }
            }

            // Always enable nodes computing output tensors or nodes whose outputs might get
            // overwritten due to inplace kernels
            // TODO (jbobba) - Do we need to handle cacheability
            if (computes_result(node.get()) || possibly_overwritten(node.get()) ||
                node->has_state())
            {
                writer << " || 1";
            }
            writer << ") {\n";
            writer.indent++;
        }

        auto it = node_function_map.find(node.get());
        if (it == node_function_map.end())
        {
            handler->second(this, writer, node.get(), in, out);
        }
        else
        {
            string func_name =
                ngraph::pass::CommonFunctionCollection::create_function_name(*it->second);
            vector<string> names;
            for (const TensorViewWrapper& tv : in)
            {
                names.push_back(tv.get_name());
            }
            for (const TensorViewWrapper& tv : out)
            {
                names.push_back(tv.get_name());
            }
            writer << func_name << "(" << join(names) << ", ctx, cg_ctx);\n";
        }

        // skip multi-output nodes since they would be covered by GetOutputElement
        if (node->get_output_size() == 1 &&
            // skip non-FP nodes
            (node->get_element_type() == element::f32 || node->get_element_type() == element::f64))
        {
            // check inputs and constants?
            if ((!node->is_parameter() && !node->is_constant()) ||
                std::getenv("NGRAPH_CPU_CHECK_PARMS_AND_CONSTS"))
            {
                if (std::getenv("NGRAPH_CPU_NAN_CHECK"))
                {
                    generate_isnan_isinf_check(writer, node, out, "isnan");
                }

                if (std::getenv("NGRAPH_CPU_INF_CHECK"))
                {
                    generate_isnan_isinf_check(writer, node, out, "isinf");
                }
            }
        }

        // Emit operation epilogue
        if (!node->is_parameter() && !node->is_constant())
        {
            for (auto output_name : node_output_names)
            {
                writer << "t_en[" << tensor_index_map[output_name] << "] = true;\n";
            }
            writer.indent--;
            writer << "} else {\n";
            writer.indent++;
            for (auto output_name : node_output_names)
            {
                writer << "t_en[" << tensor_index_map[output_name] << "] = false;\n";
            }
            writer.indent--;
            writer << "}\n";
            emit_debug_function_exit(writer, node.get(), in, out);
            if (runtime::cpu::IsTracingEnabled() && m_function->get_name() == m_function_name)
            {
                writer << "ctx->op_durations[profiler_count++] = "
                       << "(std::chrono::duration_cast<cpu::Timescale>(cpu::Clock::now() - "
                          "start_ts)).count();\n";
            }
#if defined(NGRAPH_TBB_ENABLE)
            if (m_use_tbb)
            {
                writer.indent--;
                writer << "});\n";
            }
#endif
        }
    }

#if defined(NGRAPH_TBB_ENABLE)
    if (m_use_tbb)
    {
        writer << "\n";
        // Build the flow graph

        traverse_nodes(m_function, [&writer](shared_ptr<Node> n) {
            if (!n->is_parameter() && !n->is_constant())
            {
                bool is_head = true;
                for (auto arg : n->get_arguments())
                {
                    if (!arg->is_parameter() && !arg->is_constant())
                    {
                        is_head = false;
                        writer << "tbb::flow::make_edge(*flowgraph_node_" << arg->get_name()
                               << ", *flowgraph_node_" << n->get_name() << ");\n";
                    }
                }
                if (is_head)
                {
                    writer << "tbb::flow::make_edge(*flowgraph_node_start"
                           << ", *flowgraph_node_" << n->get_name() << ");\n";
                }
            }
        });

        writer.indent--;
        writer << "}\n";

        // Execute the flow graph
        writer << "(static_cast<tbb::flow::continue_node<tbb::flow::continue_msg>*>"
                  "(&(*(cg_ctx->tbb_graph->begin()))))"
               << "->try_put(tbb::flow::continue_msg());\n";
        writer << "try { cg_ctx->tbb_graph->wait_for_all(); } catch(...) { throw; }\n";
    }
#endif
    writer << "ctx->first_iteration = false;\n";

    writer.indent--;
    // End generated function
    writer += "}\n\n";

    // TODO: Cleanup and make this a utility function
    string filename = file_util::path_join(s_output_dir, m_function_name + "_codegen.cpp");
    string code = writer.get_code();
    runtime::cpu::CPU_ExternalFunction::write_to_file(writer.get_code(), s_output_dir, filename);

    m_compiler.reset(new codegen::Compiler());
    m_execution_engine.reset(new codegen::ExecutionEngine());

    m_compiler->set_precompiled_header_source(pch_header_source);

    auto codegen_module = m_compiler->compile(code);

    if (codegen_module == nullptr)
    {
        throw runtime_error("function failed to compile");
    }
    m_execution_engine->add_module(codegen_module);
    m_execution_engine->finalize();

    m_compiled_init_ctx_func = m_execution_engine->find_function<InitContextFuncTy>("init_cg_ctx");

    if (m_compiled_init_ctx_func == nullptr)
    {
        throw runtime_error("could not find compiled init context function");
    }

    m_compiled_destroy_ctx_func =
        m_execution_engine->find_function<DestroyContextFuncTy>("destroy_cg_ctx");

    if (m_compiled_destroy_ctx_func == nullptr)
    {
        throw runtime_error("could not find compiled destroy context function");
    }

    m_compiled_function = m_execution_engine->find_function<EntryPointTy>(m_function_name);

    if (m_compiled_function == nullptr)
    {
        throw runtime_error("could not find compiled function");
    }

    // Store layouts assigned for arguments
    for (const auto& parameter : m_function->get_parameters())
    {
        for (size_t i = 0; i < parameter->get_output_size(); ++i)
        {
            auto tv = parameter->get_output_tensor_ptr(i);
            if (tv->get_tensor_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function parameter's tensor: " +
                                   tv->get_name());
            }
            parameter_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_layout()));
        }
    }

    // Store layouts assigned for results
    if (!result_layout_descriptors.empty())
    {
        throw ngraph_error("Function output layouts should not be pre-assigned");
    }
    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        const auto& output = m_function->get_output_op(i);
        for (size_t j = 0; j < output->get_output_size(); ++j)
        {
            auto tv = output->get_output_tensor_ptr(j);
            if (tv->get_tensor_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function output tensor: " + tv->get_name());
            }
            result_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_layout()));
        }
    }

    m_is_compiled = true;
    if (m_release_function && !m_emit_timing)
    {
        release_function();
    }
}

#endif // !defined(NGRAPH_DEX_ONLY)

void runtime::cpu::CPU_ExternalFunction::register_common_passes(
    ngraph::pass::Manager& pass_manager, ngraph::pass::PassConfig& pass_config)
{
    auto pass_map = pass_config.get_enables();

    auto dex = is_direct_execution();
    auto is_supported = [dex](const Node& node) {

        // this checks averts the decomposition of LSTMCell
        // we will map LSTMCell to LSTM CPU op in the later
        // graph pass
        if (typeid(ngraph::op::LSTMCell) == typeid(node))
        {
            // MKLDNN version < 1.0 doesnt support peephole for LSTM, we will skip if the LSTMCell
            // has peephole. LSTMCell with no peephole support is constant initialized to zero
            // TODO (pthoreho) : For MKLDNN > V1.0, change mkldnn kernel integration to compute for
            // LSTMCell
            // with peephole as well.
            if (is_type<ngraph::op::Constant>(node.get_argument(6)))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else if (typeid(ngraph::op::GeluBackpropFactor) == typeid(node))
        {
#if MKLDNN_VERSION_MAJOR < 1
            return ((node.input(0).get_element_type() == element::f32) ? true : false);
#else
            // TODO: will be supported in mkldnn v1.1
            return false;
#endif
        }
        else if (typeid(ngraph::op::Gelu) == typeid(node))
        {
#if MKLDNN_VERSION_MAJOR < 1
            return ((node.input(0).get_element_type() == element::f32) ? true : false);
#else
            // TODO: will be supported in mkldnn v1.1
            return false;
#endif
        }

        if (dex)
        {
            auto handler = GetGlobalBuildDispatcher().find(type_index(typeid(node)));
            if (handler == GetGlobalBuildDispatcher().end())
            {
                return false;
            }
        }
        else
        {
#if !defined(NGRAPH_DEX_ONLY)
            auto handler = dispatcher.find(type_index(typeid(node)));
            if (handler == dispatcher.end())
            {
                return false;
            }
#else
            return false;
#endif
        }
        return true;
    };

    REGISTER_KNOBBED_PASS(LikeReplacement, true, ngraph::pass)
    REGISTER_KNOBBED_PASS_WITH_ARGS(FusedOpDecomposition, true, ngraph::pass, is_supported)
    REGISTER_KNOBBED_PASS(Opset0Downgrade, true, ngraph::pass)
    REGISTER_KNOBBED_PASS(ImplicitBroadcastElimination, true, ngraph::pass)
    REGISTER_KNOBBED_PASS(NopElimination, true, ngraph::pass)
    REGISTER_KNOBBED_PASS(ZeroDimTensorElimination, true, ngraph::pass)
    REGISTER_KNOBBED_PASS(LSTMFusion, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(RNNFusion, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(AlgebraicSimplification, true, ngraph::pass)
    REGISTER_KNOBBED_PASS(MultiLayerRNNFusion, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(BiDirectionalRnn, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(CPURnnMatFusion, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(BatchFusion, true, ngraph::pass)
    REGISTER_KNOBBED_PASS(CPUBatchFusion, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(ReshapeSinking, false, ngraph::pass)
    REGISTER_KNOBBED_PASS(ReshapeElimination, true, ngraph::pass)
    REGISTER_KNOBBED_PASS(RecurrentReshapeElimination, false, ngraph::pass)
    REGISTER_KNOBBED_PASS_WITH_ARGS(
        CoreFusion, true, ngraph::pass, ngraph::pass::FusionType::ALL_FUSIONS)
    REGISTER_KNOBBED_PASS_WITH_ARGS(FusedOpDecomposition, true, ngraph::pass, is_supported)
    REGISTER_KNOBBED_PASS(CPUPreFusion, true, runtime::cpu::pass)

    // Disable CPUFusion if MLIR is enabled to preserve core ops.
    if (std::getenv("NGRAPH_MLIR") == nullptr)
    {
        REGISTER_KNOBBED_PASS(CPUFusion, true, runtime::cpu::pass)
    }
    REGISTER_KNOBBED_PASS(CPUQuantFusion, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(CPUHorizontalFusion, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(CPUCollapseDims, true, runtime::cpu::pass)
#if defined(NGRAPH_HALIDE)
    REGISTER_KNOBBED_PASS(HalideSubgraphExtraction, true, ngraph::runtime::cpu::pass)
#endif

#ifdef NGRAPH_MLIR_ENABLE
    if (std::getenv("NGRAPH_MLIR") != nullptr)
    {
        REGISTER_KNOBBED_PASS(MLIRSubgraphExtractionPass, /*enable by default*/ true, ngraph::pass)
    }
#endif

    NodeVector nv_cwi; // We dont need CPUWorkspaceInsertion to return list of indices
    REGISTER_KNOBBED_PASS_WITH_ARGS(CPUWorkspaceInsertion, true, runtime::cpu::pass, nv_cwi, false)
    REGISTER_KNOBBED_PASS_WITH_ARGS(CPUAssignment, true, runtime::cpu::pass, this)
    REGISTER_KNOBBED_PASS_WITH_ARGS(ConstantFolding, true, ngraph::pass, GetGlobalCFDispatcherCPU())
    REGISTER_KNOBBED_PASS_WITH_ARGS(CPULayout, true, runtime::cpu::pass, this)
    REGISTER_KNOBBED_PASS_WITH_ARGS(
        CommonSubexpressionElimination, true, ngraph::pass, runtime::cpu::get_cse_handlers_map())
    REGISTER_KNOBBED_PASS(CPUPostLayoutOptimizations, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(CPUConvertLayoutConstantFolding, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(CPUMemoryOptimization, true, runtime::cpu::pass)
    REGISTER_KNOBBED_PASS(GetOutputElementElimination, false, ngraph::pass)
    REGISTER_KNOBBED_PASS_WITH_ARGS(
        PropagateCacheability, true, ngraph::pass, runtime::cpu::get_annotations_factory())
    bool reuse_memory = pass_config.get_pass_attribute("CPUMemoryAssignment::ReuseMemory") ||
                        pass_config.get_pass_attribute("ReuseMemory");
    pass_manager.register_pass<runtime::cpu::pass::CPUMemoryAssignment>(
        bufferID_to_tensorSets, tensor_to_bufferID, size_t(s_memory_pool_alignment), !reuse_memory);

    pass_manager.get_state().set_visualize_tree_ops_map(runtime::cpu::get_visualize_tree_ops_map());
}

bool runtime::cpu::CPU_ExternalFunction::computes_result(Node* node)
{
    for (size_t i = 0; i < node->get_output_size(); i++)
    {
        auto& output_tensor = node->get_output_tensor(i);
        if (m_tensor_roles.find(output_tensor.get_name()) != m_tensor_roles.end() &&
            m_tensor_roles[output_tensor.get_name()] == TensorRole::OUTPUT)
        {
            return true;
        }
    }
    return false;
}

static void dump_one_kernel_with_type(runtime::cpu::CPU_DebugTracer& debug_tracer,
                                      runtime::cpu::TensorTracerAttributes& t_attrs,
                                      const std::string& kernel_name,
                                      const void* tensor,
                                      const std::string& tensor_name,
                                      const std::string& in_out)
{
    switch (t_attrs.m_type_of_element)
    {
    case element::Type_t::f32:
        debug_tracer.dump_one_tensor<float>(kernel_name,
                                            tensor,
                                            tensor_name,
                                            t_attrs.m_number_of_elements,
                                            t_attrs.m_t_shape,
                                            in_out);
        break;
    case element::Type_t::i8:
        debug_tracer.dump_one_tensor<int8_t>(kernel_name,
                                             tensor,
                                             tensor_name,
                                             t_attrs.m_number_of_elements,
                                             t_attrs.m_t_shape,
                                             in_out);
        break;
    case element::Type_t::u8:
        debug_tracer.dump_one_tensor<uint8_t>(kernel_name,
                                              tensor,
                                              tensor_name,
                                              t_attrs.m_number_of_elements,
                                              t_attrs.m_t_shape,
                                              in_out);
        break;
    case element::Type_t::i32:
        debug_tracer.dump_one_tensor<int32_t>(kernel_name,
                                              tensor,
                                              tensor_name,
                                              t_attrs.m_number_of_elements,
                                              t_attrs.m_t_shape,
                                              in_out);
        break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::boolean:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f64:
    case element::Type_t::i16:
    case element::Type_t::i64:
    case element::Type_t::u16:
    case element::Type_t::u32:
    case element::Type_t::u64:
    default: break;
    }
}

void runtime::cpu::CPU_ExternalFunction::dump_one_kernel(CPU_DebugTracer& debug_tracer,
                                                         CPURuntimeContext* ctx,
                                                         bool is_it_input)
{
    size_t index = ctx->pc;
    if (is_it_input)
    {
        for (size_t i = 0; i < m_op_attrs.at(index).Inputs.size(); i++)
        {
            dump_one_kernel_with_type(
                debug_tracer,
                m_op_attrs.at(index).m_inputs_tensor_attrs.at(i),
                m_op_attrs.at(index).Description,
                ctx->buffer_data[get_buffer_index(m_op_attrs.at(index).Inputs.at(i))],
                m_op_attrs.at(index).Inputs.at(i),
                ">>");
        }
    }
    else
    {
        for (size_t i = 0; i < m_op_attrs.at(index).Outputs.size(); i++)
        {
            dump_one_kernel_with_type(
                debug_tracer,
                m_op_attrs.at(index).m_outputs_tensor_attrs.at(i),
                m_op_attrs.at(index).Description,
                ctx->buffer_data[get_buffer_index(m_op_attrs.at(index).Outputs.at(i))],
                m_op_attrs.at(index).Outputs.at(i),
                "<<");
        }
        debug_tracer.end_of_kernel();
    }
}

void runtime::cpu::CPU_ExternalFunction::build(ngraph::pass::PassConfig& pass_config)
{
    if (m_is_built)
    {
        return;
    }

#if defined(NGRAPH_TBB_ENABLE)
    if (m_use_tbb && (runtime::cpu::IsTracingEnabled() || m_emit_timing))
    {
        throw ngraph_error(
            "CPU Backend: Tracing and performance breakdowns might not be accurate with TBB "
            "enabled due to concurrent graph execution");
    }
#endif

    // stream writer to dump the debug manifest for the DEX
    static const string s_debug_dir = "cpu_codegen";
    static StaticInitializers s_static_initializers(s_debug_dir);
    m_mkldnn_emitter.reset(new MKLDNNEmitter());
    ngraph::pass::Manager pass_manager;
    if (std::getenv("NGRAPH_ENABLE_VISUALIZE_TRACING"))
    {
        // Enable per_pass_validation if required for debug purpose
        pass_manager.set_per_pass_validation(false);
    }
    register_common_passes(pass_manager, pass_config);
    pass_manager.run_passes(m_function, false);

    static runtime::cpu::CPU_DebugTracer debug_tracer;
    if (std::getenv("NGRAPH_CPU_DEBUG_TRACER") != nullptr)
    {
        debug_tracer.set_enable_tracing(true);
    }

    // Store layouts assigned for arguments
    for (const auto& parameter : m_function->get_parameters())
    {
        for (size_t i = 0; i < parameter->get_output_size(); ++i)
        {
            auto tv = parameter->get_output_tensor_ptr(i);
            if (tv->get_tensor_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function parameter's tensor: " +
                                   tv->get_name());
            }
            parameter_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_layout()));
        }
    }

    // Store layouts assigned for results
    if (!result_layout_descriptors.empty())
    {
        throw ngraph_error("Function output layouts should not be pre-assigned");
    }
    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        const auto& output = m_function->get_output_op(i);
        for (size_t j = 0; j < output->get_output_size(); ++j)
        {
            auto tv = output->get_output_tensor_ptr(j);
            if (tv->get_tensor_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function output tensor: " + tv->get_name());
            }
            result_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_layout()));
        }
    }

    // Build executor
    size_t buffer_index = 0;
    // Temporaries
    if (m_function->get_temporary_pool_size())
    {
        m_memory_buffer_sizes.push_back(m_function->get_temporary_pool_size());
        for (auto& ele : bufferID_to_tensorSets)
        {
            if (ele.second.first == TensorRole::INTERMEDIATE)
            {
                for (auto& ele_t : ele.second.second)
                {
                    m_buffer_indices[ele_t->get_name()] = buffer_index;
                    intermediates_offsets.emplace_back(m_buffer_indices[ele_t->get_name()],
                                                       ele_t->get_pool_offset());
                    m_tensor_roles[ele_t->get_name()] = TensorRole::INTERMEDIATE;
                    buffer_index++;
                }
            }
        }
    }

    // Constants
    for (auto& node : m_function->get_ordered_ops())
    {
        if (node->is_constant())
        {
            auto output_tensor = &node->get_output_tensor();
            m_buffer_indices[output_tensor->get_name()] = buffer_index;
            constant_tensor_data.emplace_back(
                buffer_index,
                const_cast<void*>(static_pointer_cast<ngraph::op::Constant>(node)->get_data_ptr()));
            auto tensor_set = get_tensor_set(output_tensor);
            // process all tensors in the set containing the output tensor of the constant
            for (auto& ele_t : tensor_set)
            {
                NGRAPH_CHECK(ele_t->get_pool_offset() == 0, "no offset set for constants");
                m_tensor_roles[ele_t->get_name()] = TensorRole::CONSTANT;
                if (ele_t->get_name() != output_tensor->get_name())
                {
                    tensor_alias[ele_t->get_name()] = output_tensor->get_name();
                }
            }
            buffer_index++;
        }
    }

    // Inputs
    size_t arg_index = 0;
    for (auto& param : m_function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            auto output_tensor = &param->get_outputs().at(i).get_tensor();
            auto tensor_set = get_tensor_set(output_tensor);

            auto& stale = tensor_stale[output_tensor->get_name()];
            // process all tensors in the set containing the output tensor of the parameter
            for (auto& ele_t : tensor_set)
            {
                m_tensor_roles[ele_t->get_name()] = TensorRole::INPUT;
                m_buffer_indices[ele_t->get_name()] = buffer_index;
                function_input_index_offset.emplace_back(m_buffer_indices[ele_t->get_name()],
                                                         arg_index,
                                                         ele_t->get_pool_offset(),
                                                         stale);
                buffer_index++;
            }
        }
        arg_index++;
    }

    // Outputs
    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        shared_ptr<Node> op = m_function->get_output_op(i);
        auto output_tensor = &op->get_output_tensor();
        auto tensor_set = get_tensor_set(output_tensor);

        // process all tensors in the set containing the output tensor of the result
        for (auto& ele_t : tensor_set)
        {
            m_tensor_roles[ele_t->get_name()] = TensorRole::OUTPUT;
            m_buffer_indices[ele_t->get_name()] = buffer_index;
            function_output_index_offset.emplace_back(
                m_buffer_indices[ele_t->get_name()], i, ele_t->get_pool_offset());
            buffer_index++;
        }
    }

    // After processing inputs, outputs, constants, and intermediates, set the buffer size.
    m_buffer_size = buffer_index;

    for (shared_ptr<Node> node : m_function->get_ordered_ops())
    {
        if (node->is_parameter() || node->is_constant())
        {
            continue;
        }
        auto& n = *node; // Work around a compiler warning (*node inside typeid may have effects
        // with shared pointers, which is fine here but clang doesn't like it.)
        auto handler = GetGlobalBuildDispatcher().find(type_index(typeid(n)));
        if (handler == GetGlobalBuildDispatcher().end())
        {
            throw unsupported_op(node->description());
        }
        vector<TensorViewWrapper> in;
        vector<string> in_names;
        vector<TensorTracerAttributes> t_in_attrs;
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            in.push_back(TensorViewWrapper(tv, tv->get_name()));
            in_names.push_back(tv->get_name());
            t_in_attrs.push_back(TensorTracerAttributes(
                in.back().get_size(), in.back().get_shape(), in.back().get_element_type()));
        }
        vector<TensorViewWrapper> out;
        vector<string> out_names;
        vector<TensorTracerAttributes> t_out_attrs;

        for (const descriptor::Output& output : node->get_outputs())
        {
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            out.push_back(TensorViewWrapper(tv, tv->get_name()));
            out_names.push_back(tv->get_name());
            t_out_attrs.push_back(TensorTracerAttributes(
                out.back().get_size(), out.back().get_shape(), out.back().get_element_type()));
        }

        m_op_attrs.emplace_back(node->description(), out_names, in_names, t_out_attrs, t_in_attrs);
        op_names.push_back(node->get_name());
        handler->second(this, node.get(), in, out);

        auto cacheable = true;
        auto reuse_memory = pass_config.get_pass_attribute("CPUMemoryAssignment::ReuseMemory") ||
                            pass_config.get_pass_attribute("ReuseMemory");
        if (node->is_op())
        {
            auto op = std::static_pointer_cast<ngraph::op::Op>(node);
            auto op_annotations = op->get_op_annotations();
            cacheable = op_annotations->is_cacheable();
        }

        bool disable_caching =
            (reuse_memory &&
             !cacheable) // Check cacheability only if we are reusing intermediate tensors
            || computes_result(node.get()) || possibly_overwritten(node.get()) || node->has_state();

        vector<reference_wrapper<bool>> in_stale, out_stale;
        for (const auto& name : in_names)
        {
            if (tensor_alias.count(name))
            {
                in_stale.emplace_back(tensor_stale[tensor_alias[name]]);
            }
            else
            {
                in_stale.emplace_back(tensor_stale[name]);
            }
        }
        for (const auto& name : out_names)
        {
            if (tensor_alias.count(name))
            {
                out_stale.emplace_back(tensor_stale[tensor_alias[name]]);
            }
            else
            {
                out_stale.emplace_back(tensor_stale[name]);
            }
        }

        function<bool(CPURuntimeContext*)> enable;
        if (disable_caching)
        {
            enable = [in_stale, out_stale](CPURuntimeContext * /* ctx */) -> bool {
                for (auto& stale : out_stale)
                {
                    stale.get() = true;
                }
                return true;
            };
        }
        else
        {
            enable = [in_stale, out_stale](CPURuntimeContext * /* ctx */) -> bool {
                bool en = false;
                for (const auto& stale : in_stale)
                {
                    if (stale)
                    {
                        en = true;
                        break;
                    }
                }
                for (auto& stale : out_stale)
                {
                    stale.get() = en;
                }
                return en;
            };
        }

        enables.emplace_back(enable);
        enable_nodename_list.emplace_back(make_pair(enable, node->get_name()));

        m_perf_counters.emplace_back(node, 0, 0);
    }

    if ((std::getenv("NGRAPH_DEX_DEBUG") != nullptr))
    {
        string filename = file_util::path_join(s_debug_dir, m_function_name + "_debug.txt");
        std::stringstream strm;
        auto find_role = [](TensorRole tensor_role) -> string {
            switch (tensor_role)
            {
            case TensorRole::INPUT: return string("TensorRole::INPUT");
            case TensorRole::INTERMEDIATE: return string("TensorRole::INTERMEDIATE");
            case TensorRole::CONSTANT: return string("TensorRole::CONSTANT");
            case TensorRole::OUTPUT: return string("TensorRole::OUTPUT");
            case TensorRole::UNKNOWN:
            default: throw runtime_error("unhandled CPU tensor role");
            }
        };

        // dump the tensor roles to debug manifest
        for (const auto& tensor_roles : m_tensor_roles)
        {
            strm << tensor_roles.first << ", " << find_role(tensor_roles.second) << "\n";
        }

        write_to_file(strm.str(), s_debug_dir, filename);
        strm.str("");

        // dump the op's order of execution along with the address of tensor_data which holds the
        // base address of each tensor.
        for (shared_ptr<Node> node : m_function->get_ordered_ops())
        {
            std::vector<string> node_inputs;
            std::vector<string> node_outputs;
            std::stringstream temp;
            if (node->is_parameter() || node->is_constant())
            {
                continue;
            }
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                temp << &m_buffer_indices[tv->get_name()];
                node_inputs.push_back(tv->get_name() + "(" + temp.str() + ")");
                temp.str("");
            }

            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                temp << &m_buffer_indices[tv->get_name()];
                node_outputs.push_back(tv->get_name() + "(" + temp.str() + ")");
                temp.str("");
            }
            strm << "\n" << node->get_name() << "(";
            vector<string> parameter_nodes = node_inputs;
            parameter_nodes.insert(parameter_nodes.end(), node_outputs.begin(), node_outputs.end());
            strm << join(parameter_nodes);
            strm << ")\n";
            write_to_file(strm.str(), s_debug_dir, filename);
            strm.str("");
        }
    }
    // This check ensures we have exactly one functor for Op.
    NGRAPH_CHECK(m_op_attrs.size() == functors.size());

    executor = [&](CPURuntimeContext* ctx, vector<void*>& inputs, vector<void*>& outputs) {
        cpu::Timestamp start_ts, end_ts;
        uint64_t profiler_count = 0;

        if (ctx->first_iteration)
        {
            for (auto& p : intermediates_offsets)
            {
                ctx->buffer_data[p.first] =
                    static_cast<uint8_t*>(ctx->memory_buffers[0]->get_ptr()) + p.second;
            }

            for (auto& p : constant_tensor_data)
            {
                ctx->buffer_data[p.first] = p.second;
            }
        }

        for (const auto& p : function_input_index_offset)
        {
            ctx->buffer_data[get<0>(p)] = static_cast<uint8_t*>(inputs[get<1>(p)]) + get<2>(p);
            get<3>(p).get() = ctx->p_en[get<1>(p)];
        }

        for (const auto& p : function_output_index_offset)
        {
            ctx->buffer_data[get<0>(p)] = static_cast<uint8_t*>(outputs[get<1>(p)]) + get<2>(p);
        }

        auto functor = functors.begin();
#if defined(NGRAPH_TBB_ENABLE)
        if (m_use_tbb)
        {
            // Build the flow graph
            if (ctx->first_iteration)
            {
                std::unordered_map<std::string, tbb::flow::continue_node<tbb::flow::continue_msg>*>
                    nodename_tbbnode_map;
                tbb::flow::continue_node<tbb::flow::continue_msg>* flowgraph_node_start =
                    new tbb::flow::continue_node<tbb::flow::continue_msg>(
                        *(ctx->G), [&](const tbb::flow::continue_msg& /* msg */) {});
                auto it = enable_nodename_list.begin();
                for (const auto& p : enables)
                {
                    auto index = profiler_count++;
                    tbb::flow::continue_node<tbb::flow::continue_msg>* flowgraph_node =
                        new tbb::flow::continue_node<tbb::flow::continue_msg>(
                            *(ctx->G),
                            [&, functor, index](const tbb::flow::continue_msg& /* msg */) {
                                if (p(ctx) || ctx->first_iteration)
                                {
                                    if (runtime::cpu::IsTracingEnabled() || m_emit_timing)
                                    {
                                        start_ts = cpu::Clock::now();
                                    }
                                    CPUExecutionContext ectx{0};
                                    executor::GetCPUExecutor().execute(*functor, ctx, &ectx, true);
                                    if (runtime::cpu::IsTracingEnabled() || m_emit_timing)
                                    {
                                        end_ts = cpu::Clock::now();

                                        if (runtime::cpu::IsTracingEnabled())
                                        {
                                            ctx->op_durations[index] =
                                                (std::chrono::duration_cast<cpu::Timescale>(
                                                     end_ts - start_ts))
                                                    .count();
                                        }
                                        if (m_emit_timing)
                                        {
                                            m_perf_counters[index].m_total_microseconds +=
                                                std::chrono::duration_cast<
                                                    std::chrono::microseconds>(end_ts - start_ts)
                                                    .count();
                                            m_perf_counters[index].m_call_count++;
                                        }
                                    }
                                }
                                else
                                {
                                    if (runtime::cpu::IsTracingEnabled())
                                    {
                                        ctx->op_durations[index] = 0;
                                    }
                                    if (m_emit_timing)
                                    {
                                        m_perf_counters[index].m_call_count++;
                                    }
                                }
                            });
#ifdef TBB_PREVIEW_FLOW_GRAPH_TRACE
                    flowgraph_node->set_name(it->second.c_str());
#endif
                    std::advance(functor, 1);
                    nodename_tbbnode_map.insert({it->second, flowgraph_node});
                    it++;
                }

                traverse_nodes(
                    m_function, [&flowgraph_node_start, &nodename_tbbnode_map](shared_ptr<Node> n) {
                        if (!n->is_parameter() && !n->is_constant())
                        {
                            bool is_head = true;
                            for (auto arg : n->get_arguments())
                            {
                                if (!arg->is_parameter() && !arg->is_constant())
                                {
                                    is_head = false;
                                    tbb::flow::make_edge(*(nodename_tbbnode_map[arg->get_name()]),
                                                         *(nodename_tbbnode_map[n->get_name()]));
                                }
                            }
                            if (is_head)
                            {
                                tbb::flow::make_edge(*flowgraph_node_start,
                                                     *(nodename_tbbnode_map[n->get_name()]));
                            }
                        }
                    });

                if (m_release_function)
                {
                    release_function();
                }
            }
            // Execute the flow graph
            (static_cast<tbb::flow::continue_node<tbb::flow::continue_msg>*>(&(*(ctx->G->begin()))))
                ->try_put(tbb::flow::continue_msg());
            try
            {
                ctx->G->wait_for_all();
            }
            catch (...)
            {
                throw;
            }
        }
        else
#endif
        {
            static const auto ddebug = std::getenv("NGRAPH_DEX_DEBUG");
            if (ddebug != nullptr)
            {
                if (ctx->first_iteration)
                {
                    string filename =
                        file_util::path_join(s_debug_dir, m_function_name + "_debug.txt");
                    std::stringstream ss;

                    ss << "\nEXECUTION PLAN:\n";

                    for (size_t i = 0; i < functors.size(); i++)
                    {
                        ss << op_names.at(i) << " will be executed with the following inputs:\n";
                        for (auto& is : this->m_op_attrs.at(i).Inputs)
                        {
                            ss << "\t" << is << " = "
                               << ctx->buffer_data[this->get_buffer_index(is)] << std::endl;
                        }
                        ss << "and outputs :\n";
                        for (auto& os : this->m_op_attrs.at(i).Outputs)
                        {
                            ss << "\t" << os << " = "
                               << ctx->buffer_data[this->get_buffer_index(os)] << std::endl;
                        }
                    }
                    write_to_file(ss.str(), s_debug_dir, filename);
                }
            }

            for (; ctx->pc < functors.size(); ctx->pc++)
            {
                auto index = profiler_count++;
                if ((enables.at(ctx->pc))(ctx) || ctx->first_iteration)
                {
                    // Each Op will have exactly one functor, start the clock before the exceution
                    // of functor
                    // and collect the profiler_count once the execution complets
                    if (runtime::cpu::IsTracingEnabled() || m_emit_timing)
                    {
                        start_ts = cpu::Clock::now();
                    }

                    CPUExecutionContext ectx{0};

                    if (debug_tracer.tracing_is_enabled())
                    {
                        this->dump_one_kernel(debug_tracer, ctx, true);
                    }

                    executor::GetCPUExecutor().execute(functors.at(ctx->pc), ctx, &ectx);

                    if (debug_tracer.tracing_is_enabled())
                    {
                        this->dump_one_kernel(debug_tracer, ctx, false);
                    }

                    if (ctx->breakpoints.count(ctx->pc + 1))
                    {
                        ctx->pc++;
                        break;
                    }

                    if (runtime::cpu::IsTracingEnabled() || m_emit_timing)
                    {
                        end_ts = cpu::Clock::now();

                        if (runtime::cpu::IsTracingEnabled())
                        {
                            ctx->op_durations[index] =
                                (std::chrono::duration_cast<cpu::Timescale>(end_ts - start_ts))
                                    .count();
                        }
                        if (m_emit_timing)
                        {
                            m_perf_counters[index].m_total_microseconds +=
                                std::chrono::duration_cast<std::chrono::microseconds>(end_ts -
                                                                                      start_ts)
                                    .count();
                            m_perf_counters[index].m_call_count++;
                        }
                    }
                }
                else
                {
                    if (runtime::cpu::IsTracingEnabled())
                    {
                        ctx->op_durations[index] = 0;
                    }
                    if (m_emit_timing)
                    {
                        m_perf_counters[index].m_call_count++;
                    }
                }
            }
        }
        ctx->first_iteration = false;
        if (runtime::cpu::IsTracingEnabled())
        {
            NGRAPH_CHECK(m_op_attrs.size() == profiler_count);
        }

    };

    m_is_built = true;

#if defined(NGRAPH_TBB_ENABLE)
    if (m_release_function && !m_use_tbb)
#else
    if (m_release_function)
#endif
    {
        release_function();
    }
}

size_t runtime::cpu::CPU_ExternalFunction::get_buffer_index(const std::string& name)
{
    if (tensor_alias.count(name))
    {
        NGRAPH_CHECK(m_buffer_indices.count(tensor_alias[name]));
        return m_buffer_indices[tensor_alias[name]];
    }
    else
    {
        NGRAPH_CHECK(m_buffer_indices.count(name));
        return m_buffer_indices[name];
    }
}

bool runtime::cpu::CPU_ExternalFunction::is_codegen(const ngraph::pass::PassConfig& pc)
{
    auto attrs = pc.get_pass_attributes();
    auto it = attrs.find("CODEGEN");
    if (it != attrs.end())
    {
        return it->second;
    }
    return false;
}

shared_ptr<ngraph::runtime::cpu::CPU_CallFrame>
    runtime::cpu::CPU_ExternalFunction::make_call_frame(ngraph::pass::PassConfig& pass_config,
                                                        Allocator* allocator)
{
#if defined(NGRAPH_DEX_ONLY)
    if (is_codegen(pass_config))
    {
        NGRAPH_WARN << "CPU Backend: Requested unsupported compilation mode (CODEGEN). Falling "
                       "back to DEX instead";
    }
#else
    // Override DEX if pass_config requests CODEGEN
    if (is_codegen(pass_config))
    {
        m_direct_execution = false;
    }
    if (!m_is_compiled && !m_direct_execution)
    {
        compile(pass_config);
    }
#endif

    if (!m_is_built && m_direct_execution)
    {
        build(pass_config);
    }

    return make_shared<ngraph::runtime::cpu::CPU_CallFrame>(shared_from_this(),
                                                            m_compiled_init_ctx_func,
                                                            m_compiled_destroy_ctx_func,
                                                            m_compiled_function,
                                                            allocator);
}

const runtime::cpu::LayoutDescriptorPtrs&
    runtime::cpu::CPU_ExternalFunction::get_parameter_layout_descriptors()
{
    return parameter_layout_descriptors;
}

const runtime::cpu::LayoutDescriptorPtrs&
    runtime::cpu::CPU_ExternalFunction::get_result_layout_descriptors()
{
    return result_layout_descriptors;
}

const vector<runtime::PerformanceCounter>& runtime::cpu::CPU_ExternalFunction::get_perf_counters()
{
#if !defined(NGRAPH_DEX_ONLY)
    // Codegen. Retrieve perf counters from compiled module
    if (m_execution_engine)
    {
        auto get_count = m_execution_engine->find_function<size_t()>("get_debug_timer_count");
        auto get_name =
            m_execution_engine->find_function<const char*(size_t)>("get_debug_timer_name");
        auto get_microseconds =
            m_execution_engine->find_function<size_t(size_t)>("get_debug_timer_microseconds");
        auto get_call_count =
            m_execution_engine->find_function<size_t(size_t)>("get_debug_timer_call_count");

        if (get_count && get_name && get_microseconds && get_call_count)
        {
            size_t count = get_count();
            if (m_perf_counters.size() == 0)
            {
                map<string, shared_ptr<const Node>> name_map;
                for (auto n : m_function->get_ops())
                {
                    name_map.insert({n->get_name(), n});
                }
                for (size_t i = 0; i < count; i++)
                {
                    shared_ptr<const Node> n = name_map[get_name(i)];
                    m_perf_counters.push_back({n, get_microseconds(i), get_call_count(i)});
                }
            }
            else
            {
                for (size_t i = 0; i < count; i++)
                {
                    m_perf_counters[i].m_total_microseconds = get_microseconds(i);
                    m_perf_counters[i].m_call_count = get_call_count(i);
                }
            }
        }
        if (m_release_function)
        {
            release_function();
        }
    }
#endif
    return m_perf_counters;
}

void runtime::cpu::CPU_ExternalFunction::write_to_file(const std::string& code,
                                                       const std::string& directory,
                                                       const std::string& filename)
{
    std::ofstream out;
    file_util::make_directory(directory);
    bool is_exist = file_util::exists(filename);
    is_exist ? out.open(filename, std::ofstream::app) : out.open(filename);
    out << code;
    out.close();
}

#if !defined(NGRAPH_DEX_ONLY)

void runtime::cpu::CPU_ExternalFunction::emit_debug_function_entry(
    CodeWriter& writer,
    Node* node,
    const std::vector<TensorViewWrapper>& /* in */,
    const std::vector<TensorViewWrapper>& /* out */)
{
    if (m_emit_timing)
    {
        writer << "timers[" << m_name_index_map[node->get_name()] << "].start();\n";
    }
}

void runtime::cpu::CPU_ExternalFunction::emit_debug_function_exit(
    CodeWriter& writer,
    Node* node,
    const std::vector<TensorViewWrapper>& /* in */,
    const std::vector<TensorViewWrapper>& /* out */)
{
    if (m_emit_timing)
    {
        writer << "timers[" << m_name_index_map[node->get_name()] << "].stop();\n";
    }
}

bool runtime::cpu::CPU_ExternalFunction::is_functionally_identical(
    const Node& n1, const Node& n2, const unordered_map<const Node*, string>& node_cache)
{
    return node_cache.at(&n1) == node_cache.at(&n2);
}

string runtime::cpu::CPU_ExternalFunction::emit_op_as_function(const Node& node,
                                                               const string& function_name)
{
    CodeWriter writer;
    writer << "static void " << function_name << "(";
    writer.indent++;
    // Work around a compiler warning (*node inside typeid may have effects
    // with shared pointers, which is fine here but clang doesn't like it.)
    auto handler = dispatcher.find(type_index(typeid(node)));
    if (handler == dispatcher.end())
    {
        throw unsupported_op(node.description());
    }
    vector<TensorViewWrapper> in;
    size_t arg_index = 0;
    set<string> arg_names;
    for (const descriptor::Input& input : node.get_inputs())
    {
        const descriptor::Output& output = input.get_output();
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        TensorViewWrapper tvw{tv, "_arg" + to_string(arg_index)};
        if (arg_names.find(tvw.get_name()) == arg_names.end())
        {
            arg_names.insert(tvw.get_name());
            if (arg_index++ > 0)
            {
                writer << ",";
            }
            writer << "\n";
            writer << tvw.get_type() << "* " << tvw.get_name();
        }
        in.push_back(tvw);
    }
    vector<TensorViewWrapper> out;
    for (const descriptor::Output& output : node.get_outputs())
    {
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        TensorViewWrapper tvw{tv, "_out" + to_string(arg_index)};
        if (arg_index++ > 0)
        {
            writer << ",";
        }
        writer << "\n";
        writer << tvw.get_type() << "* " << tvw.get_name();
        out.push_back(tvw);
    }
    writer << ",\ncpu::CPURuntimeContext* ctx, CPURuntimeContextCG* cg_ctx";
    writer.indent--;
    writer << "\n)\n";
    writer << "{\n";
    writer.indent++;
    handler->second(this, writer, &node, in, out);
    writer.indent--;
    writer << "}\n";

    string rc = writer.get_code();
    if (function_name == "f")
    {
        rc = strip_comments(rc);
    }
    return rc;
}

string runtime::cpu::CPU_ExternalFunction::strip_comments(const string& s)
{
    stringstream out;
    for (size_t i = 0; i < s.size(); i++)
    {
        if (i < s.size() - 2)
        {
            if (s[i] == '/' && s[i + 1] == '/')
            {
                // line comment
                i += 2;
                while (s[i] != '\n')
                {
                    i++;
                }
                out << '\n';
            }
            else if (s[i] == '/' && s[i + 1] == '*')
            {
                // multi-line comment
                i += 2;
                while (!(s[i] == '*' && s[i + 1] == '/'))
                {
                    i++;
                }
                i++;
            }
            else
            {
                out << s[i];
            }
        }
        else
        {
            out << s[i];
        }
    }
    return out.str();
}

#endif

std::unordered_set<descriptor::Tensor*>&
    runtime::cpu::CPU_ExternalFunction::get_tensor_set(descriptor::Tensor* output_tensor)
{
    auto output_tensor_it = tensor_to_bufferID.find(output_tensor);
    NGRAPH_CHECK(output_tensor_it != tensor_to_bufferID.end());
    auto bufferID = output_tensor_it->second;
    auto output_buffer_it = bufferID_to_tensorSets.find(bufferID);
    NGRAPH_CHECK(output_buffer_it != bufferID_to_tensorSets.end());
    return output_buffer_it->second.second;
}
