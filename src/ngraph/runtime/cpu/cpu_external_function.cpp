//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#define TBB_PREVIEW_FLOW_GRAPH_TRACE 1

#include <tbb/flow_graph.h>

#if !defined(NGRAPH_DEX_ONLY)
#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
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
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
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
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/function_call.hpp"
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
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/select_and_scatter.hpp"
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
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/any_all_replacement.hpp"
#include "ngraph/pass/common_function_collection.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/propagate_cacheability.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/reshape_sinking.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_cse.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"
#include "ngraph/runtime/cpu/cpu_visualize_tree.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_dot.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/loop_kernel.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"
#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"
#include "ngraph/runtime/cpu/pass/cpu_collapse_dims.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_horizontal_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_layout.hpp"
#include "ngraph/runtime/cpu/pass/cpu_mat_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_memory_optimization.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"
#include "ngraph/runtime/cpu/pass/cpu_rnn_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_workspace_insertion.hpp"
#include "ngraph/runtime/cpu/pass/halide_subgraph_extraction.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/op/allreduce.hpp"
#endif

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
    , m_use_tbb(std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
#if !defined(NGRAPH_DEX_ONLY)
    , m_is_compiled(false)
    , m_direct_execution(!std::getenv("NGRAPH_CODEGEN"))
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
#ifdef NGRAPH_DISTRIBUTED
    {TI(ngraph::op::AllReduce), &runtime::cpu::CPU_Emitter::emit<op::AllReduce>},
#endif
    {TI(ngraph::op::MatmulBias), &runtime::cpu::CPU_Emitter::emit<op::MatmulBias>},
    {TI(ngraph::op::Dot), &runtime::cpu::CPU_Emitter::emit<op::Dot>},
    {TI(ngraph::op::Multiply), &runtime::cpu::CPU_Emitter::emit<op::Multiply>},
    {TI(ngraph::op::Parameter), &runtime::cpu::CPU_Emitter::nop},
    {TI(ngraph::op::Abs), &runtime::cpu::CPU_Emitter::emit<op::Abs>},
    {TI(ngraph::op::BatchDot), &runtime::cpu::CPU_Emitter::emit<op::BatchDot>},
    {TI(ngraph::op::Concat), &runtime::cpu::CPU_Emitter::emit<op::Concat>},
    {TI(ngraph::op::Divide), &runtime::cpu::CPU_Emitter::emit<op::Divide>},
    {TI(ngraph::op::Equal), &runtime::cpu::CPU_Emitter::emit<op::Equal>},
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
    {TI(ngraph::op::FunctionCall), &runtime::cpu::CPU_Emitter::emit<op::FunctionCall>},
    {TI(ngraph::op::Reduce), &runtime::cpu::CPU_Emitter::emit<op::Reduce>},
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
    {TI(ngraph::op::QuantizedMaxPool), &runtime::cpu::CPU_Emitter::emit<op::QuantizedMaxPool>},
    {TI(ngraph::op::QuantizedAvgPool), &runtime::cpu::CPU_Emitter::emit<op::QuantizedAvgPool>},
    {TI(ngraph::op::MaxPoolWithIndices), &runtime::cpu::CPU_Emitter::emit<op::MaxPoolWithIndices>},
    {TI(ngraph::op::Reverse), &runtime::cpu::CPU_Emitter::emit<op::Reverse>},
    {TI(ngraph::op::ReverseSequence), &runtime::cpu::CPU_Emitter::emit<op::ReverseSequence>},
    {TI(ngraph::op::Result), &runtime::cpu::CPU_Emitter::emit<op::Result>},
    {TI(ngraph::op::ReduceWindow), &runtime::cpu::CPU_Emitter::emit<op::ReduceWindow>},
    {TI(ngraph::op::SelectAndScatter), &runtime::cpu::CPU_Emitter::emit<op::SelectAndScatter>},
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
    {TI(ngraph::op::LeakyRelu), &runtime::cpu::CPU_Emitter::emit<op::LeakyRelu>},
    {TI(ngraph::runtime::cpu::op::LoopKernel),
     &runtime::cpu::CPU_Emitter::emit<runtime::cpu::op::LoopKernel>},
    {TI(ngraph::op::LRN), &runtime::cpu::CPU_Emitter::emit<ngraph::op::LRN>},
    {TI(ngraph::op::GenerateMask), &runtime::cpu::CPU_Emitter::emit<ngraph::op::GenerateMask>},
    {TI(ngraph::op::ConvolutionAdd), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionAdd>},
    {TI(ngraph::op::Quantize), &runtime::cpu::CPU_Emitter::emit<ngraph::op::Quantize>},
    {TI(ngraph::op::Dequantize), &runtime::cpu::CPU_Emitter::emit<ngraph::op::Dequantize>},
    {TI(ngraph::op::GroupConvolutionBias),
     &runtime::cpu::CPU_Emitter::emit<op::GroupConvolutionBias>},
};

static void
    generate_isnan_isinf_check(codegen::CodeWriter& writer,
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

void runtime::cpu::CPU_ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    m_mkldnn_emitter.reset(new MKLDNNEmitter());

    ngraph::pass::Manager pass_manager;
    register_common_passes(pass_manager);
    unordered_map<Node*, Node*> node_function_map;
    string common_function_string;
    auto femitter = bind(&ngraph::runtime::cpu::CPU_ExternalFunction::emit_op_as_function,
                         this,
                         placeholders::_1,
                         placeholders::_2);
    pass_manager.register_pass<ngraph::pass::CommonFunctionCollection>(
        femitter, node_function_map, common_function_string);
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::PropagateCacheability>(
        runtime::cpu::get_annotations_factory());
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(size_t(s_memory_pool_alignment), true);
    pass_manager.run_passes(m_function);

    unordered_map<shared_ptr<Function>, list<shared_ptr<Node>>> function_ordered_ops;
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        function_ordered_ops.insert({current_function, current_function->get_ordered_ops()});
    }

    codegen::CodeWriter writer;

    writer << "// Generated by the nGraph CPU backend\n";
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

#ifdef NGRAPH_DISTRIBUTED
    writer << "#include <mlsl.hpp>\n";
    writer << "#define NGRAPH_DISTRIBUTED\n";
#endif

    writer +=
        R"(
#include <cmath>
#include "ngraph/except.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_eigen_utils.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/reference/and.hpp"
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
#include "ngraph/runtime/reference/reduce.hpp"
#include "ngraph/runtime/reference/reduce_window.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/select_and_scatter.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/topk.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/state/rng_state.hpp"
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
        for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
        {
            for (shared_ptr<Node> node : function_ordered_ops.at(current_function))
            {
                if (!node->is_parameter() && !node->is_constant())
                {
                    names.push_back(node->get_name());
                    m_name_index_map.insert({node->get_name(), index++});
                }
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
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        for (shared_ptr<Node> node : function_ordered_ops.at(current_function))
        {
            ngraph::op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
            if (c)
            {
                m_active_constants.push_back(node);
                shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                string type = tv->get_element_type().c_type_string();
                writer << "static " << type << "* " << tv->get_name() << " = ((" << type << "*)("
                       << c->get_data_ptr() << "));\n";
                m_variable_name_map[tv->get_name()] = tv->get_name();
                m_tensor_roles[tv->get_name()] = CPUTensorRole::CONSTANT;
            }
        }
    }

    writer << "// Declare all functions\n";
    for (shared_ptr<Function> f : pass_manager.get_state().get_functions())
    {
        writer << "extern \"C\" void " << f->get_name()
               << "(void** inputs, void** outputs, cpu::CPURuntimeContext* ctx);\n";
    }
    writer << "\n";

    writer << common_function_string << "\n";

    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        auto ordered_ops = function_ordered_ops.at(current_function);
        set<string> output_names;
        for (shared_ptr<Node> op : current_function->get_results())
        {
            shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
            output_names.insert(tv->get_name());
        }
        set<descriptor::Tensor*> constants;
        for (shared_ptr<Node> node : ordered_ops)
        {
            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                constants.insert(tv.get());
            }
        }

        bool temporaries_used = false;
        size_t worst_case_tmp_size = 0;
        for (shared_ptr<Node> node : ordered_ops)
        {
            if (node->liveness_new_list.size() > 0)
            {
                temporaries_used = true;
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    worst_case_tmp_size += tensor->size();
                }
            }
        }
        if (temporaries_used)
        {
            m_memory_buffer_sizes.push_back(current_function->get_temporary_pool_size());
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

        // In place concatenation optimization
        process_in_place_concat(ordered_ops);

        writer << "bool " << current_function->get_name() << "_t_en[" << tensor_index << "];\n";

        writer << "extern \"C\" void " << current_function->get_name();
        writer << "(void** inputs, void** outputs, cpu::CPURuntimeContext* ctx)\n";
        writer << "{\n";
        writer.indent++;

        // Execution tracing support
        if (runtime::cpu::IsTracingEnabled() && current_function->get_name() == m_function_name)
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

        writer << "bool* t_en = (bool*)" << current_function->get_name() << "_t_en;\n";

        if (m_use_tbb)
        {
            writer << "\n";
            writer << "if (ctx->first_iteration) {\n";
            writer.indent++;
            writer << "tbb::flow::continue_node<tbb::flow::continue_msg>* "
                      "flowgraph_node_start"
                   << " = new tbb::flow::continue_node<tbb::flow::continue_msg> "
                      "(*(ctx->G), [&](const tbb::flow::continue_msg &msg)\n{});\n";
        }

        for (shared_ptr<Node> node : ordered_ops)
        {
            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                propagate_in_place_constant(&node->get_outputs().at(0), tv->get_name(), false);
            }
        }

        // Add inputs to the variable name map
        size_t arg_index = 0;
        for (shared_ptr<ngraph::op::Parameter> param : current_function->get_parameters())
        {
            for (size_t i = 0; i < param->get_output_size(); ++i)
            {
                shared_ptr<descriptor::Tensor> tv = param->get_output_tensor_ptr(i);
                const element::Type& et = tv->get_element_type();
                string type = et.c_type_string();
                stringstream ss;
                ss << "((" << type << "*)(inputs[" << arg_index << "]))";
                m_variable_name_map[tv->get_name()] = ss.str();
                m_tensor_roles[tv->get_name()] = CPUTensorRole::INPUT;
                param_index_map[tv->get_name()] = arg_index;
                propagate_in_place_input(&param->get_outputs().at(i), ss.str(), false);
                arg_index++;
            }
        }

        // In place slice optimization
        process_in_place_slice(ordered_ops);

        if (temporaries_used)
        {
            // Add temporaries to the variable name map
            for (shared_ptr<Node> node : ordered_ops)
            {
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    stringstream ss;
                    ss << "((" << tensor->get_element_type().c_type_string()
                       << "*)(pool_base_ptr + " << tensor->get_pool_offset() << "))";
                    if (m_tensor_roles.find(tensor->get_name()) == m_tensor_roles.end())
                    {
                        m_variable_name_map[tensor->get_name()] = ss.str();
                        m_tensor_roles[tensor->get_name()] = CPUTensorRole::INTERMEDIATE;
                    }
                }
            }
        }

        // Add outputs to the variable name map
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
            string type = tv->get_element_type().c_type_string();
            stringstream ss;
            ss << "((" << type << "*)(outputs[" << i << "]))";
            m_variable_name_map[tv->get_name()] = ss.str();
            m_tensor_roles[tv->get_name()] = CPUTensorRole::OUTPUT;

            //keep assigning different outputs to a result descriptor
            //op::Result emitter will check if in and out descriptors are the same
            //and skip a copy
            auto res = std::dynamic_pointer_cast<ngraph::op::Result>(op);
            auto input_node = res->get_inputs().at(0).get_output().get_node();
            if (!input_node->is_constant() && !input_node->is_parameter())
            {
                shared_ptr<descriptor::Tensor> itv =
                    res->get_inputs().at(0).get_output().get_tensor_ptr();
                auto output_name = ss.str();
                m_variable_name_map[itv->get_name()] = ss.str();
                m_tensor_roles[itv->get_name()] = CPUTensorRole::OUTPUT;
                propagate_in_place_output(
                    &(res->get_inputs().at(0).get_output()), output_name, false);
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
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                in.push_back(TensorViewWrapper(tv, m_variable_name_map[tv->get_name()]));
                node_input_names.emplace_back(tv->get_name());
            }
            vector<TensorViewWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                out.push_back(TensorViewWrapper(tv, m_variable_name_map[tv->get_name()]));
                node_output_names.emplace_back(tv->get_name());
            }

            // Emit operation prologue
            if (!node->is_parameter() && !node->is_constant())
            {
                if (current_function->get_name() == m_function_name)
                {
                    m_op_attrs.emplace_back(
                        node->description(), node_output_names, node_input_names);
                }
                if (m_use_tbb)
                {
                    writer << "tbb::flow::continue_node<tbb::flow::continue_msg>* "
                              "flowgraph_node_"
                           << node->get_name()
                           << " = new tbb::flow::continue_node<tbb::flow::continue_msg> "
                              "(*(ctx->G), [&](const tbb::flow::continue_msg &msg)\n{\n";
                    writer.indent++;
                }
                if (runtime::cpu::IsTracingEnabled() &&
                    current_function->get_name() == m_function_name)
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
                if (computes_result(node.get()) || possibly_overwritten(node.get()))
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
                writer << func_name << "(" << join(names) << ", ctx);\n";
            }

            // skip multi-output nodes since they would be covered by GetOutputElement
            if (node->get_output_size() == 1 &&
                // skip non-FP nodes
                (node->get_element_type() == element::f32 ||
                 node->get_element_type() == element::f64))
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
                if (runtime::cpu::IsTracingEnabled() &&
                    current_function->get_name() == m_function_name)
                {
                    writer << "ctx->op_durations[profiler_count++] = "
                           << "(std::chrono::duration_cast<cpu::Timescale>(cpu::Clock::now() - "
                              "start_ts)).count();\n";
                }
                if (m_use_tbb)
                {
                    writer.indent--;
                    writer << "});\n";
                }
            }
        }

        if (m_use_tbb)
        {
            writer << "\n";
            // Build the flow graph

            traverse_nodes(current_function, [&writer](shared_ptr<Node> n) {
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
                      "(&(*(ctx->G->begin()))))"
                   << "->try_put(tbb::flow::continue_msg());\n";
            writer << "try { ctx->G->wait_for_all(); } catch(...) { throw; }\n";
        }
        writer << "ctx->first_iteration = false;\n";

        writer.indent--;
        // End generated function
        writer += "}\n\n";
    }

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
    m_compiled_function = m_execution_engine->find_function<EntryPoint_t>(m_function_name);

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
                throw ngraph_error("layout missing on function parameter's tensor view: " +
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
    if (m_release_function)
    {
        release_function();
    }
}

#endif // !defined(NGRAPH_DEX_ONLY)

void runtime::cpu::CPU_ExternalFunction::register_common_passes(ngraph::pass::Manager& pass_manager)
{
    auto pass_map = pass_manager.get_pass_config().get_enables();

    REGISTER_KNOBBED_PASS(AnyAllReplacement, true, ngraph::pass);
    REGISTER_KNOBBED_PASS(LikeReplacement, true, ngraph::pass);
    REGISTER_KNOBBED_PASS(NopElimination, true, ngraph::pass);
    REGISTER_KNOBBED_PASS(ZeroDimTensorElimination, true, ngraph::pass);
    REGISTER_KNOBBED_PASS(LSTMFusion, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(RNNFusion, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(AlgebraicSimplification, true, ngraph::pass);
    REGISTER_KNOBBED_PASS(MultiLayerRNNFusion, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(CPURnnMatFusion, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(CPUBatchFusion, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(ReshapeSinking, false, ngraph::pass);
    REGISTER_KNOBBED_PASS(ReshapeElimination, false, ngraph::pass);
    REGISTER_KNOBBED_PASS(CoreFusion, true, ngraph::pass);
    REGISTER_KNOBBED_PASS(CPUFusion, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(CPUHorizontalFusion, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(CPUCollapseDims, true, runtime::cpu::pass);
#if defined(NGRAPH_HALIDE)
    REGISTER_KNOBBED_PASS(HalideSubgraphExtraction, true, ngraph::runtime::cpu::pass);
#endif

    NodeVector nv_cwi; // We dont need CPUWorkspaceInsertion to return list of indices
    REGISTER_KNOBBED_PASS_WITH_ARGS(CPUWorkspaceInsertion, true, runtime::cpu::pass, nv_cwi, false);
    REGISTER_KNOBBED_PASS_WITH_ARGS(CPUAssignment, true, runtime::cpu::pass, this);
    REGISTER_KNOBBED_PASS(ConstantFolding, false, ngraph::pass);
    REGISTER_KNOBBED_PASS_WITH_ARGS(CPULayout, true, runtime::cpu::pass, this);
    REGISTER_KNOBBED_PASS_WITH_ARGS(
        CommonSubexpressionElimination, true, ngraph::pass, runtime::cpu::get_cse_handlers_map());
    REGISTER_KNOBBED_PASS(CPUPostLayoutOptimizations, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(CPUMemoryOptimization, true, runtime::cpu::pass);
    REGISTER_KNOBBED_PASS(GetOutputElementElimination, false, ngraph::pass);
    pass_manager.get_state().set_visualize_tree_ops_map(runtime::cpu::get_visualize_tree_ops_map());
}

bool runtime::cpu::CPU_ExternalFunction::computes_result(Node* node)
{
    for (size_t i = 0; i < node->get_output_size(); i++)
    {
        auto& output_tensor = node->get_output_tensor(i);
        if (m_tensor_roles.find(output_tensor.get_name()) != m_tensor_roles.end() &&
            m_tensor_roles[output_tensor.get_name()] == CPUTensorRole::OUTPUT)
        {
            return true;
        }
    }
    return false;
}

void runtime::cpu::CPU_ExternalFunction::propagate_in_place_input(
    ngraph::descriptor::Output* output, std::string input_name, bool dex)
{
    std::deque<ngraph::descriptor::Output*> stack;
    stack.push_front(output);

    while (stack.size() > 0)
    {
        ngraph::descriptor::Output* it = stack.front();
        stack.pop_front();
        for (auto input : it->get_inputs())
        {
            auto input_node = input->get_node();
            if (!input_node->is_op() || input_node->is_output() ||
                dynamic_pointer_cast<ngraph::op::Slice>(input_node))
            {
                continue;
            }

            auto c_op = std::static_pointer_cast<ngraph::op::Op>(input_node);
            if (auto op_annotations = c_op->get_op_annotations())
            {
                for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                {
                    if (oi_pair.input == input->get_index() && !oi_pair.destructive)
                    {
                        size_t output_index = oi_pair.output;
                        auto& output_tensor = c_op->get_outputs().at(output_index).get_tensor();

                        if (dex)
                        {
                            tensor_alias[output_tensor.get_name()] = input_name;
                        }
                        else
                        {
                            m_variable_name_map[output_tensor.get_name()] = input_name;
                        }
                        m_tensor_roles[output_tensor.get_name()] = CPUTensorRole::INPUT;

                        NGRAPH_DEBUG << "CPU codegen: Forwarding " << input_name << " through "
                                     << output_tensor.get_name();
                        stack.push_back(&c_op->get_outputs().at(output_index));
                    }
                }
            }
        }
    }
}

void runtime::cpu::CPU_ExternalFunction::propagate_in_place_constant(
    ngraph::descriptor::Output* output, std::string input_name, bool dex)
{
    std::deque<ngraph::descriptor::Output*> stack;
    stack.push_front(output);

    while (stack.size() > 0)
    {
        ngraph::descriptor::Output* it = stack.front();
        stack.pop_front();
        for (auto input : it->get_inputs())
        {
            auto input_node = input->get_node();
            if (!input_node->is_op() || input_node->is_output())
            {
                continue;
            }

            auto c_op = std::static_pointer_cast<ngraph::op::Op>(input_node);
            if (auto op_annotations = c_op->get_op_annotations())
            {
                for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                {
                    if (oi_pair.input == input->get_index() && !oi_pair.destructive)
                    {
                        size_t output_index = oi_pair.output;
                        auto& output_tensor = c_op->get_outputs().at(output_index).get_tensor();

                        if (dex)
                        {
                            tensor_alias[output_tensor.get_name()] = input_name;
                        }
                        else
                        {
                            m_variable_name_map[output_tensor.get_name()] = input_name;
                        }
                        m_tensor_roles[output_tensor.get_name()] = CPUTensorRole::CONSTANT;

                        NGRAPH_DEBUG << " CPU: Forwarding " << input_name << " through "
                                     << output_tensor.get_name();
                        stack.push_back(&c_op->get_outputs().at(output_index));
                    }
                }
            }
        }
    }
}

void runtime::cpu::CPU_ExternalFunction::propagate_in_place_output(
    ngraph::descriptor::Output* res_src_output, std::string output_name, bool dex)
{
    // we start with a particular output
    // which is an argument to a given op::Result
    size_t offset = res_src_output->get_tensor().get_pool_offset();
    auto it = res_src_output;

    bool propagate_further = false;
    do
    {
        propagate_further = false;
        auto it_node = it->get_node();

        if (!it_node->is_op() || std::dynamic_pointer_cast<ngraph::op::Slice>(it_node))
        {
            break;
        }

        auto arg = std::static_pointer_cast<ngraph::op::Op>(it_node);
        if (auto op_annotations = arg->get_op_annotations())
        {
            for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
            {
                if (oi_pair.output == it->get_index())
                {
                    size_t input_index = oi_pair.input;
                    auto& input_tensor = arg->get_inputs().at(input_index).get_tensor();
                    auto tmp_node = arg->get_inputs().at(input_index).get_output().get_node();
                    if (input_tensor.get_pool_offset() == offset && !tmp_node->is_parameter() &&
                        !tmp_node->is_constant())
                    {
                        NGRAPH_DEBUG << "Reusing " << output_name << " for "
                                     << input_tensor.get_name();

                        if (dex)
                        {
                            tensor_alias[input_tensor.get_name()] = output_name;
                        }
                        else
                        {
                            m_variable_name_map[input_tensor.get_name()] = output_name;
                        }
                        m_tensor_roles[input_tensor.get_name()] = CPUTensorRole::OUTPUT;

                        it = &arg->get_inputs().at(input_index).get_output();
                        propagate_further = true;
                    }
                }
            }
        }
    } while (propagate_further);
}

void runtime::cpu::CPU_ExternalFunction::process_in_place_concat(
    std::list<std::shared_ptr<Node>> nodes)
{
    for (shared_ptr<Node> node : nodes)
    {
        if (auto concat = std::dynamic_pointer_cast<ngraph::op::Concat>(node))
        {
            if (auto op_annotations = concat->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    auto output_tensor = &concat->get_output_tensor();
                    auto offset = output_tensor->get_pool_offset();
                    for (auto arg : concat->get_arguments())
                    {
                        auto input_tensor = &arg->get_output_tensor();
                        auto old_offset = input_tensor->get_pool_offset();
                        input_tensor->set_pool_offset(offset);
                        NGRAPH_DEBUG << "cpu_external_function: change offset, old offset is "
                                     << old_offset << ", new offset is " << offset << std::endl;
                        offset += input_tensor->size();
                    }

                    bool found_last_concat = true;
                    for (auto user : concat->get_users())
                    {
                        if (auto user_concat = dynamic_pointer_cast<ngraph::op::Concat>(user))
                        {
                            if (auto user_op_annotations = user_concat->get_op_annotations())
                            {
                                auto user_in_place_oi_pairs =
                                    user_op_annotations->get_in_place_oi_pairs();
                                if (user_in_place_oi_pairs.size() > 0)
                                {
                                    found_last_concat = false;
                                    break;
                                }
                            }
                        }
                    }

                    if (found_last_concat)
                    {
                        for (auto arg : concat->get_arguments())
                        {
                            if (auto arg_concat = dynamic_pointer_cast<ngraph::op::Concat>(arg))
                            {
                                NGRAPH_DEBUG
                                    << "cpu_external_function: call propagate_in_place_concat for "
                                    << arg->get_name() << std::endl;
                                propagate_in_place_concat(arg_concat);
                            }
                        }
                    }
                }
            }
        }
    }
}

void runtime::cpu::CPU_ExternalFunction::propagate_in_place_concat(
    shared_ptr<ngraph::op::Concat> concat)
{
    std::deque<std::shared_ptr<ngraph::op::Concat>> stack;
    stack.push_front(concat);

    while (stack.size() > 0)
    {
        auto it = stack.front();
        stack.pop_front();
        if (auto op_annotations = it->get_op_annotations())
        {
            auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
            if (in_place_oi_pairs.size() > 0)
            {
                auto output_tensor = &it->get_output_tensor();
                auto offset = output_tensor->get_pool_offset();
                for (auto arg : it->get_arguments())
                {
                    auto input_tensor = &arg->get_output_tensor();
                    auto old_offset = input_tensor->get_pool_offset();
                    input_tensor->set_pool_offset(offset);
                    NGRAPH_DEBUG
                        << "cpu_external_function, propagate: concat, change offset, old offset is "
                        << old_offset << ", new offset is " << offset << std::endl;
                    offset += input_tensor->size();
                    if (auto arg_concat = std::dynamic_pointer_cast<ngraph::op::Concat>(arg))
                    {
                        stack.push_front(arg_concat);
                    }
                }
            }
        }
    }
}

//slice
void runtime::cpu::CPU_ExternalFunction::process_in_place_slice(
    std::list<std::shared_ptr<Node>> nodes)
{
    for (shared_ptr<Node>& node : nodes)
    {
        if (auto slice = std::dynamic_pointer_cast<ngraph::op::Slice>(node))
        {
            if (auto op_annotations = slice->get_op_annotations())
            {
                auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                if (in_place_oi_pairs.size() > 0)
                {
                    auto input = &slice->get_inputs().at(0);
                    auto arg = input->get_output().get_node();
                    auto index = input->get_output().get_index();
                    auto input_tensor = &arg->get_output_tensor(index);
                    if (m_tensor_roles.find(input_tensor->get_name()) != m_tensor_roles.end() &&
                        m_tensor_roles[input_tensor->get_name()] == CPUTensorRole::INPUT)
                    {
                        NGRAPH_DEBUG << "cpu_external_function: function input pointer passed to "
                                        "slice, do not change offset.";
                        continue;
                    }
                    auto offset = input_tensor->get_pool_offset();
                    auto lower_bounds = slice->get_lower_bounds();
                    auto start = 0, accumulated = 1;
                    auto in_shape = slice->get_input_shape(0);
                    for (int i = in_shape.size() - 1; i >= 0; i--)
                    {
                        start += lower_bounds[i] * accumulated;
                        accumulated *= in_shape[i];
                    }
                    offset += node->get_element_type().size() * start;
                    auto output_tensor = &slice->get_output_tensor();
                    auto old_offset = output_tensor->get_pool_offset();

                    output_tensor->set_pool_offset(offset);
                    NGRAPH_DEBUG << "cpu_external_function: slice, change offset, old offset is "
                                 << old_offset << ", new offset is " << offset;
                }
            }
        }
    }
}

void runtime::cpu::CPU_ExternalFunction::build()
{
    if (m_is_built)
    {
        return;
    }

    if (m_use_tbb && (runtime::cpu::IsTracingEnabled() || m_emit_timing))
    {
        throw ngraph_error(
            "CPU Backend: Tracing and performance breakdowns might not be accurate with TBB "
            "enabled due to concurrent graph execution");
    }

    // stream writer to dump the debug manifest for the DEX
    static const string s_debug_dir = "cpu_codegen";
    static StaticInitializers s_static_initializers(s_debug_dir);
    m_mkldnn_emitter.reset(new MKLDNNEmitter());
    ngraph::pass::Manager pass_manager;
    register_common_passes(pass_manager);
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::PropagateCacheability>(
        runtime::cpu::get_annotations_factory());
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(size_t(s_memory_pool_alignment), true);
    pass_manager.run_passes(m_function, false);

    // Store layouts assigned for arguments
    for (const auto& parameter : m_function->get_parameters())
    {
        for (size_t i = 0; i < parameter->get_output_size(); ++i)
        {
            auto tv = parameter->get_output_tensor_ptr(i);
            if (tv->get_tensor_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function parameter's tensor view: " +
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

    // In place concatenation optimization
    process_in_place_concat(m_function->get_ordered_ops());

    // Constants
    for (auto& node : m_function->get_ordered_ops())
    {
        if (node->is_constant())
        {
            auto tv = node->get_outputs()[0].get_tensor_ptr();
            tensor_data[tv->get_name()] =
                const_cast<void*>(static_pointer_cast<ngraph::op::Constant>(node)->get_data_ptr());
            m_tensor_roles[tv->get_name()] = CPUTensorRole::CONSTANT;
            propagate_in_place_constant(&node->get_outputs().at(0), tv->get_name(), true);
        }
    }

    // Inputs
    size_t arg_index = 0;
    for (auto& param : m_function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            shared_ptr<descriptor::Tensor> tv = param->get_output_tensor_ptr(i);
            function_input_index.emplace_back(
                tensor_data[tv->get_name()], arg_index, tensor_stale[tv->get_name()]);
            m_tensor_roles[tv->get_name()] = CPUTensorRole::INPUT;
            propagate_in_place_input(&param->get_outputs().at(i), tv->get_name(), true);
            arg_index++;
        }
    }

    // In place slice optimization
    process_in_place_slice(m_function->get_ordered_ops());

    // Intermediates
    if (m_function->get_temporary_pool_size())
    {
        m_memory_buffer_sizes.push_back(m_function->get_temporary_pool_size());

        for (auto& node : m_function->get_ordered_ops())
        {
            for (auto tensor : node->liveness_new_list)
            {
                if (m_tensor_roles.find(tensor->get_name()) == m_tensor_roles.end())
                {
                    intermediates_offsets.emplace_back(tensor_data[tensor->get_name()],
                                                       tensor->get_pool_offset());
                    m_tensor_roles[tensor->get_name()] = CPUTensorRole::INTERMEDIATE;
                }
            }
        }
    }

    // Outputs
    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        shared_ptr<Node> op = m_function->get_output_op(i);
        shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
        function_output_index.emplace_back(tensor_data[tv->get_name()], i);
        m_tensor_roles[tv->get_name()] = CPUTensorRole::OUTPUT;

        //keep assigning different outputs to a result descriptor
        //op::Result emitter will check if in and out descriptors are the same
        //and skip a copy
        auto res = std::dynamic_pointer_cast<ngraph::op::Result>(op);
        auto input_node = res->get_inputs().at(0).get_output().get_node();
        if (!input_node->is_constant() && !input_node->is_parameter())
        {
            shared_ptr<descriptor::Tensor> itv =
                res->get_inputs().at(0).get_output().get_tensor_ptr();
            function_output_index.emplace_back(tensor_data[itv->get_name()], i);
            m_tensor_roles[itv->get_name()] = CPUTensorRole::OUTPUT;
            tensor_alias[itv->get_name()] = tv->get_name();
            propagate_in_place_output(
                &(res->get_inputs().at(0).get_output()), tv->get_name(), true);
        }
    }

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
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            in.push_back(TensorViewWrapper(tv, tv->get_name()));
            in_names.push_back(tv->get_name());
        }
        vector<TensorViewWrapper> out;
        vector<string> out_names;

        for (const descriptor::Output& output : node->get_outputs())
        {
            shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
            out.push_back(TensorViewWrapper(tv, tv->get_name()));
            out_names.push_back(tv->get_name());
        }

        m_op_attrs.emplace_back(node->description(), out_names, in_names);
        op_names.push_back(node->get_name());
        handler->second(this, node.get(), in, out);

        bool disable_caching = computes_result(node.get()) || possibly_overwritten(node.get());

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
            out_stale.emplace_back(tensor_stale[name]);
        }

        function<bool(CPURuntimeContext*)> enable;
        if (disable_caching)
        {
            enable = [in_stale, out_stale](CPURuntimeContext* ctx) -> bool {
                for (auto& stale : out_stale)
                {
                    stale.get() = true;
                }
                return true;
            };
        }
        else
        {
            enable = [in_stale, out_stale](CPURuntimeContext* ctx) -> bool {
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

        m_perf_counters.emplace_back(node->get_name().c_str(), 0, 0);
    }

    if ((std::getenv("NGRAPH_DEX_DEBUG") != nullptr))
    {
        string filename = file_util::path_join(s_debug_dir, m_function_name + "_debug.txt");
        std::stringstream strm;
        auto find_role = [](CPUTensorRole tensor_role) -> string {
            switch (tensor_role)
            {
            case CPUTensorRole::INPUT: return string("CPUTensorRole::INPUT");
            case CPUTensorRole::INTERMEDIATE: return string("CPUTensorRole::INTERMEDIATE");
            case CPUTensorRole::CONSTANT: return string("CPUTensorRole::CONSTANT");
            case CPUTensorRole::OUTPUT: return string("CPUTensorRole::OUTPUT");
            }
            throw runtime_error("unhandled CPU tensor role");
        };

        //dump the tensor roles to debug manifest
        for (const auto& tensor_roles : m_tensor_roles)
        {
            strm << tensor_roles.first << ", " << find_role(tensor_roles.second) << "\n";
        }

        write_to_file(strm.str(), s_debug_dir, filename);
        strm.str("");

        //dump the op's order of execution along with the address of
        //tensor_data which holds the base address of each tensor.
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
                temp << &tensor_data[tv->get_name()];
                node_inputs.push_back(tv->get_name() + "(" + temp.str() + ")");
                temp.str("");
            }

            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                temp << &tensor_data[tv->get_name()];
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
    //This check ensures we have exactly one functor for Op.
    assert(m_op_attrs.size() == functors.size());

    executor = [&](CPURuntimeContext* ctx, vector<void*>& inputs, vector<void*>& outputs) {
        cpu::Timestamp start_ts, end_ts;
        int profiler_count = 0;

        if (ctx->first_iteration)
        {
            for (auto& p : intermediates_offsets)
            {
                p.first.get() = static_cast<uint8_t*>(ctx->memory_buffers[0]->get_ptr()) + p.second;
            }
        }

        for (const auto& p : function_input_index)
        {
            get<0>(p).get() = inputs[get<1>(p)];
            get<2>(p).get() = ctx->p_en[get<1>(p)];
        }

        for (const auto& p : function_output_index)
        {
            p.first.get() = outputs[p.second];
        }

        auto functor = functors.begin();
        if (m_use_tbb)
        {
            // Build the flow graph
            if (ctx->first_iteration)
            {
                std::unordered_map<std::string, tbb::flow::continue_node<tbb::flow::continue_msg>*>
                    nodename_tbbnode_map;
                tbb::flow::continue_node<tbb::flow::continue_msg>* flowgraph_node_start =
                    new tbb::flow::continue_node<tbb::flow::continue_msg>(
                        *(ctx->G), [&](const tbb::flow::continue_msg& msg) {});
                auto it = enable_nodename_list.begin();
                for (const auto& p : enables)
                {
                    auto index = profiler_count++;
                    tbb::flow::continue_node<tbb::flow::continue_msg>* flowgraph_node =
                        new tbb::flow::continue_node<tbb::flow::continue_msg>(
                            *(ctx->G), [&, functor, index](const tbb::flow::continue_msg& msg) {
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
        {
            static const auto ddebug = std::getenv("NGRAPH_DEX_DEBUG");
            if (ddebug != nullptr)
            {
                if (ctx->first_iteration)
                {
                    string filename =
                        file_util::path_join(s_debug_dir, m_function_name + "_debug.txt");
                    std::stringstream ss;

                    ss << "EXECUTION PLAN:\n";
                    for (size_t i = 0; i < functors.size(); i++)
                    {
                        ss << op_names.at(i) << "will be executed with the following inputs:\n";
                        for (auto is : this->m_op_attrs.at(i).Inputs)
                        {
                            ss << "\t" << is << " = " << this->get_tensor_data(is) << std::endl;
                        }
                        ss << "and outputs :\n";
                        for (auto os : this->m_op_attrs.at(i).Outputs)
                        {
                            ss << "\t" << os << " = " << this->get_tensor_data(os) << std::endl;
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
                    // Each Op will have exactly one functor, start the clock before the exceution of functor
                    // and collect the profiler_count once the execution complets
                    if (runtime::cpu::IsTracingEnabled() || m_emit_timing)
                    {
                        start_ts = cpu::Clock::now();
                    }
                    CPUExecutionContext ectx{0};
                    executor::GetCPUExecutor().execute(functors.at(ctx->pc), ctx, &ectx);
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
            assert(m_op_attrs.size() == profiler_count);
        }

    };

    m_is_built = true;

    if (m_release_function && !m_use_tbb)
    {
        release_function();
    }
}

void*& runtime::cpu::CPU_ExternalFunction::get_tensor_data(const std::string& name)
{
    if (tensor_alias.count(name))
    {
        return tensor_data[tensor_alias[name]];
    }
    else
    {
        return tensor_data[name];
    }
}

shared_ptr<ngraph::runtime::cpu::CPU_CallFrame>
    runtime::cpu::CPU_ExternalFunction::make_call_frame()
{
#if !defined(NGRAPH_DEX_ONLY)
    if (!m_is_compiled && !m_direct_execution)
    {
        compile();
    }
#endif

    if (!m_is_built && m_direct_execution)
    {
        build();
    }

    return make_shared<ngraph::runtime::cpu::CPU_CallFrame>(shared_from_this(),
                                                            m_compiled_function);
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
                for (size_t i = 0; i < count; i++)
                {
                    m_perf_counters.push_back(
                        {get_name(i), get_microseconds(i), get_call_count(i)});
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
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<TensorViewWrapper>& in,
    const std::vector<TensorViewWrapper>& out)
{
    if (m_emit_timing)
    {
        writer << "timers[" << m_name_index_map[node->get_name()] << "].start();\n";
    }
}

void runtime::cpu::CPU_ExternalFunction::emit_debug_function_exit(
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<TensorViewWrapper>& in,
    const std::vector<TensorViewWrapper>& out)
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
    codegen::CodeWriter writer;
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
    writer << ",\ncpu::CPURuntimeContext* ctx";
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
