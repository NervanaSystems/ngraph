/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

// Kill clang diagnostics bug
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-id-macro"

#undef __TBB_PREVIEW_LIGHTWEIGHT_POLICY
#define __TBB_PREVIEW_LIGHTWEIGHT_POLICY 1

#pragma clang diagnostic pop

#include <tbb/flow_graph.h>

#if !defined(NGRAPH_DEX_ONLY)
#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#endif

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
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
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
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
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/remainder.hpp"
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
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/common_function_collection.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/result_copy_elimination.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_dot.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/loop_kernel.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"
#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"
#include "ngraph/runtime/cpu/pass/cpu_collapse_dims.hpp"
#include "ngraph/runtime/cpu/pass/cpu_concat_inputs.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_layout.hpp"
#include "ngraph/runtime/cpu/pass/cpu_mat_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_post_layout_optimizations.hpp"
#include "ngraph/runtime/cpu/pass/cpu_rnn_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_workspace_insertion.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/op/allreduce.hpp"
#endif

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_ExternalFunction::CPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function, bool release_function)
    : m_function(function)
    , m_release_function(release_function)
    , m_use_tbb(std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
    , m_compiled_function(nullptr)
#if !defined(NGRAPH_DEX_ONLY)
    , m_is_compiled(false)
    , m_emit_timing(false)
#endif
    , m_function_name(function->get_name())
    , m_is_built(false)
#if !defined(NGRAPH_DEX_ONLY)
    , m_direct_execution(std::getenv("NGRAPH_DEX") != nullptr)
#else
    , m_direct_execution(true)
#endif
{
}

runtime::cpu::CPU_ExternalFunction::~CPU_ExternalFunction()
{
}

#if !defined(NGRAPH_DEX_ONLY)

static const string s_output_dir = "cpu_codegen";

class StaticInitializers
{
public:
    StaticInitializers() { ngraph::file_util::remove_directory(s_output_dir); }
};

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

static StaticInitializers s_static_initializers;

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
    {TI(ngraph::op::Exp), &runtime::cpu::CPU_Emitter::emit<op::Exp>},
    {TI(ngraph::op::Sin), &runtime::cpu::CPU_Emitter::emit<op::Sin>},
    {TI(ngraph::op::Sinh), &runtime::cpu::CPU_Emitter::emit<op::Sinh>},
    {TI(ngraph::op::Cos), &runtime::cpu::CPU_Emitter::emit<op::Cos>},
    {TI(ngraph::op::Cosh), &runtime::cpu::CPU_Emitter::emit<op::Cosh>},
    {TI(ngraph::op::Tan), &runtime::cpu::CPU_Emitter::emit<op::Tan>},
    {TI(ngraph::op::Tanh), &runtime::cpu::CPU_Emitter::emit<op::Tanh>},
    {TI(ngraph::op::Asin), &runtime::cpu::CPU_Emitter::emit<op::Asin>},
    {TI(ngraph::op::ArgMin), &runtime::cpu::CPU_Emitter::emit<op::ArgMin>},
    {TI(ngraph::op::ArgMax), &runtime::cpu::CPU_Emitter::emit<op::ArgMax>},
    {TI(ngraph::op::Acos), &runtime::cpu::CPU_Emitter::emit<op::Acos>},
    {TI(ngraph::op::Atan), &runtime::cpu::CPU_Emitter::emit<op::Atan>},
    {TI(ngraph::op::ReplaceSlice), &runtime::cpu::CPU_Emitter::emit<op::ReplaceSlice>},
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
    {TI(ngraph::op::ConvolutionRelu), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionRelu>},
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
    {TI(ngraph::op::ReduceWindow), &runtime::cpu::CPU_Emitter::emit<op::ReduceWindow>},
    {TI(ngraph::op::SelectAndScatter), &runtime::cpu::CPU_Emitter::emit<op::SelectAndScatter>},
    {TI(ngraph::op::AvgPool), &runtime::cpu::CPU_Emitter::emit<op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop), &runtime::cpu::CPU_Emitter::emit<op::AvgPoolBackprop>},
    {TI(ngraph::op::Pad), &runtime::cpu::CPU_Emitter::emit<op::Pad>},
    {TI(ngraph::op::BatchNorm), &runtime::cpu::CPU_Emitter::emit<op::BatchNorm>},
    {TI(ngraph::op::BatchNormRelu), &runtime::cpu::CPU_Emitter::emit<op::BatchNormRelu>},
    {TI(ngraph::op::BatchNormBackprop), &runtime::cpu::CPU_Emitter::emit<op::BatchNormBackprop>},
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
    {TI(ngraph::runtime::cpu::op::LoopKernel),
     &runtime::cpu::CPU_Emitter::emit<runtime::cpu::op::LoopKernel>},
    {TI(ngraph::op::LRN), &runtime::cpu::CPU_Emitter::emit<ngraph::op::LRN>},
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

    // nv_cwi is required only by some frontends
    // in which case they should run this pass(CPUWorkspaceInsertion) explicitly
    NodeVector nv_cwi;
    pass_manager.register_pass<ngraph::pass::NopElimination>();
    // TODO (pruthvi): Enable all the disabeled RNN fusion graph pass after fixing
    // failing mxnet unit tests.
    // pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    // pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    // pass_manager.register_pass<runtime::cpu::pass::MultiLayerRNNFusion>();
    // pass_manager.register_pass<runtime::cpu::pass::ConcatInputs>();
    pass_manager.register_pass<runtime::cpu::pass::CPUBatchFusion>();
    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<ngraph::pass::CoreFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUCollapseDims>();
    pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);
    pass_manager.register_pass<runtime::cpu::pass::CPUAssignment>(this);
    pass_manager.register_pass<runtime::cpu::pass::CPULayout>(this);
    pass_manager.register_pass<runtime::cpu::pass::CPUPostLayoutOptimizations>();
    pass_manager.register_pass<ngraph::pass::ResultCopyElimination>();
    pass_manager.register_pass<ngraph::pass::GetOutputElementElimination>();
    unordered_map<Node*, Node*> node_function_map;
    string common_function_string;
    auto femitter = bind(&ngraph::runtime::cpu::CPU_ExternalFunction::emit_op_as_function,
                         this,
                         placeholders::_1,
                         placeholders::_2);
    pass_manager.register_pass<ngraph::pass::CommonFunctionCollection>(
        femitter, node_function_map, common_function_string);
    pass_manager.register_pass<ngraph::pass::Liveness>();
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
        writer << "#undef __TBB_PREVIEW_LIGHTWEIGHT_POLICY \n";
        writer << "#define __TBB_PREVIEW_LIGHTWEIGHT_POLICY 1\n";
        writer << "#include <tbb/flow_graph.h>";
    }

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
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/product.hpp"
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
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/util.hpp"

using namespace ngraph::runtime::cpu::eigen;
using namespace ngraph::runtime;

)";

#ifdef NGRAPH_DISTRIBUTED
    writer << "#include <mpi.h>\n\n";
#endif

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
            const ngraph::op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
            if (c)
            {
                m_active_constants.push_back(node);
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                string type = tv->get_tensor().get_element_type().c_type_string();
                writer << "static " << type << "* " << tv->get_tensor().get_name() << " = (("
                       << type << "*)(" << c->get_data_ptr() << "));\n";
                m_variable_name_map[tv->get_tensor().get_name()] = tv->get_tensor().get_name();
                m_tensor_roles[tv->get_tensor().get_name()] = CPUTensorRole::CONSTANT;
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
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            output_names.insert(tv->get_tensor().get_name());
        }
        set<descriptor::TensorView*> constants;
        for (shared_ptr<Node> node : ordered_ops)
        {
            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
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
                    shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                    tensor_index_map.insert({tv->get_tensor().get_name(), tensor_index++});
                }
            }
        }

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

            // Add temporaries to the variable name map
            for (shared_ptr<Node> node : ordered_ops)
            {
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    stringstream ss;
                    ss << "((" << tensor->get_element_type().c_type_string()
                       << "*)(pool_base_ptr + " << tensor->get_pool_offset() << "))";
                    m_variable_name_map[tensor->get_name()] = ss.str();
                    m_tensor_roles[tensor->get_name()] = CPUTensorRole::INTERMEDIATE;
                }
            }
        }

        writer << "bool* t_en = (bool*)" << current_function->get_name() << "_t_en;\n";

        if (m_use_tbb)
        {
            writer << "\n";
            writer << "if (ctx->first_iteration) {\n";
            writer.indent++;
            writer << "tbb::flow::continue_node<tbb::flow::continue_msg, tbb::flow::lightweight>* "
                      "flowgraph_node_start"
                   << " = new tbb::flow::continue_node<tbb::flow::continue_msg, "
                      "tbb::flow::lightweight>"
                      "(*(ctx->G), [&](const tbb::flow::continue_msg &msg)\n{});\n";
        }

        // Add inputs to the variable name map
        size_t arg_index = 0;
        for (shared_ptr<ngraph::op::Parameter> param : current_function->get_parameters())
        {
            for (size_t i = 0; i < param->get_output_size(); ++i)
            {
                shared_ptr<descriptor::TensorView> tv = param->get_output_tensor_view(i);
                const element::Type& et = tv->get_tensor_view_type()->get_element_type();
                string type = et.c_type_string();
                stringstream ss;
                ss << "((" << type << "*)(inputs[" << arg_index << "]))";
                m_variable_name_map[tv->get_tensor().get_name()] = ss.str();
                m_tensor_roles[tv->get_tensor().get_name()] = CPUTensorRole::INPUT;
                param_index_map[tv->get_tensor().get_name()] = arg_index;
                propagate_in_place_input(&param->get_outputs().at(i), ss.str(), false);
                arg_index++;
            }
        }

        // Add outputs to the variable name map
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            string type = tv->get_tensor_view_type()->get_element_type().c_type_string();
            stringstream ss;
            ss << "((" << type << "*)(outputs[" << i << "]))";
            m_variable_name_map[tv->get_tensor().get_name()] = ss.str();
            m_tensor_roles[tv->get_tensor().get_name()] = CPUTensorRole::OUTPUT;

            // it should be safe to assign both descriptors to one output*
            // since needs_copy == false makes `op::Result` an nop
            auto res = std::dynamic_pointer_cast<ngraph::op::Result>(op);
            if (!res->needs_copy())
            {
                shared_ptr<descriptor::TensorView> itv =
                    res->get_inputs().at(0).get_output().get_tensor_view();

                auto output_name = ss.str();
                m_variable_name_map[itv->get_tensor().get_name()] = ss.str();
                m_tensor_roles[itv->get_tensor().get_name()] = CPUTensorRole::OUTPUT;
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
                throw ngraph_error("Unhandled op during code generation : " + node->description());
            }
            vector<TensorViewWrapper> in;
            vector<string> node_input_names;
            vector<string> node_output_names;
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                in.push_back(
                    TensorViewWrapper(tv, m_variable_name_map[tv->get_tensor().get_name()]));
                node_input_names.emplace_back(tv->get_tensor().get_name());
            }
            vector<TensorViewWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                out.push_back(
                    TensorViewWrapper(tv, m_variable_name_map[tv->get_tensor().get_name()]));
                node_output_names.emplace_back(tv->get_tensor().get_name());
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
                    writer << "tbb::flow::continue_node<tbb::flow::continue_msg, "
                              "tbb::flow::lightweight>* "
                              "flowgraph_node_"
                           << node->get_name()
                           << " = new tbb::flow::continue_node<tbb::flow::continue_msg, "
                              "tbb::flow::lightweight>"
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
                    shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                    auto input_name = tv->get_tensor().get_name();

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
            writer << "(static_cast<tbb::flow::continue_node<tbb::flow::continue_msg, "
                      "tbb::flow::lightweight>*>(&(*(ctx->G->begin()))))"
                   << "->try_put(tbb::flow::continue_msg());\n";
            writer << "try { ctx->G->wait_for_all(); } catch(...) { throw; }\n";
        }
        writer << "ctx->first_iteration = false;\n";

        writer.indent--;
        // End generated function
        writer += "}\n\n";
    }

    // TODO: Cleanup and make this a utility function
    file_util::make_directory(s_output_dir);
    string filename = file_util::path_join(s_output_dir, m_function_name + "_codegen.cpp");
    ofstream out(filename);
    string code = writer.get_code();
    out << code;
    out.close();

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
            auto tv = parameter->get_output_tensor_view(i);
            if (tv->get_tensor_view_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function parameter's tensor view: " +
                                   tv->get_name());
            }
            parameter_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_view_layout()));
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
            auto tv = output->get_output_tensor_view(j);
            if (tv->get_tensor_view_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function output tensor: " + tv->get_name());
            }
            result_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_view_layout()));
        }
    }

    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

#endif

bool runtime::cpu::CPU_ExternalFunction::computes_result(Node* node)
{
    for (size_t i = 0; i < node->get_output_size(); i++)
    {
        auto& output_tensor = node->get_output_tensor(i);
        if (m_tensor_roles[output_tensor.get_name()] == CPUTensorRole::OUTPUT)
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
            auto c_op = std::dynamic_pointer_cast<ngraph::op::Op>(input->get_node());
            if (!c_op || c_op->is_output())
            {
                continue;
            }

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
        auto arg = std::dynamic_pointer_cast<ngraph::op::Op>(it->get_node());
        if (!arg)
        {
            break;
        }
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

void runtime::cpu::CPU_ExternalFunction::build()
{
    if (m_is_built)
    {
        return;
    }

    m_mkldnn_emitter.reset(new MKLDNNEmitter());

    ngraph::pass::Manager pass_manager;

    // nv_cwi is required only by some frontends
    // in which case they should run this pass(CPUWorkspaceInsertion) explicitly
    NodeVector nv_cwi;
    pass_manager.register_pass<ngraph::pass::NopElimination>();
    // TODO (pruthvi): Enable all the disabeled RNN fusion graph pass after fixing
    // failing mxnet unit tests.
    // pass_manager.register_pass<runtime::cpu::pass::LSTMFusion>();
    // pass_manager.register_pass<runtime::cpu::pass::RNNFusion>();
    // pass_manager.register_pass<runtime::cpu::pass::ConcatInputs>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<runtime::cpu::pass::CPUBatchFusion>();
    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<ngraph::pass::CoreFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUCollapseDims>();
    pass_manager.register_pass<runtime::cpu::pass::CPUWorkspaceInsertion>(nv_cwi);
    pass_manager.register_pass<runtime::cpu::pass::CPUAssignment>(this);
    pass_manager.register_pass<runtime::cpu::pass::CPULayout>(this);
    pass_manager.register_pass<runtime::cpu::pass::CPUPostLayoutOptimizations>();
    pass_manager.register_pass<ngraph::pass::ResultCopyElimination>();
    pass_manager.register_pass<ngraph::pass::GetOutputElementElimination>();
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(size_t(s_memory_pool_alignment), true);
    pass_manager.run_passes(m_function, false);

    // Store layouts assigned for arguments
    for (const auto& parameter : m_function->get_parameters())
    {
        for (size_t i = 0; i < parameter->get_output_size(); ++i)
        {
            auto tv = parameter->get_output_tensor_view(i);
            if (tv->get_tensor_view_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function parameter's tensor view: " +
                                   tv->get_name());
            }
            parameter_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_view_layout()));
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
            auto tv = output->get_output_tensor_view(j);
            if (tv->get_tensor_view_layout() == nullptr)
            {
                throw ngraph_error("layout missing on function output tensor: " + tv->get_name());
            }
            result_layout_descriptors.emplace_back(
                static_pointer_cast<runtime::cpu::LayoutDescriptor>(tv->get_tensor_view_layout()));
        }
    }

    // Build executor
    // Intermediates
    if (m_function->get_temporary_pool_size())
    {
        m_memory_buffer_sizes.push_back(m_function->get_temporary_pool_size());

        for (auto& node : m_function->get_ordered_ops())
        {
            for (auto tensor : node->liveness_new_list)
            {
                intermediates_offsets.emplace_back(tensor_data[tensor->get_name()],
                                                   tensor->get_pool_offset());
                m_tensor_roles[tensor->get_name()] = CPUTensorRole::INTERMEDIATE;
            }
        }
    }

    // Constants
    for (auto& node : m_function->get_ordered_ops())
    {
        if (node->is_constant())
        {
            auto tv = node->get_outputs()[0].get_tensor_view();
            tensor_data[tv->get_tensor().get_name()] =
                const_cast<void*>(static_pointer_cast<ngraph::op::Constant>(node)->get_data_ptr());
            m_tensor_roles[tv->get_tensor().get_name()] = CPUTensorRole::CONSTANT;
        }
    }

    // Inputs
    size_t arg_index = 0;
    for (auto& param : m_function->get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            shared_ptr<descriptor::TensorView> tv = param->get_output_tensor_view(i);
            function_input_index.emplace_back(tensor_data[tv->get_tensor().get_name()],
                                              arg_index,
                                              tensor_stale[tv->get_tensor().get_name()]);
            m_tensor_roles[tv->get_tensor().get_name()] = CPUTensorRole::INPUT;
            propagate_in_place_input(
                &param->get_outputs().at(i), tv->get_tensor().get_name(), true);
            arg_index++;
        }
    }

    // Outputs
    for (size_t i = 0; i < m_function->get_output_size(); ++i)
    {
        shared_ptr<Node> op = m_function->get_output_op(i);
        shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
        function_output_index.emplace_back(tensor_data[tv->get_tensor().get_name()], i);
        m_tensor_roles[tv->get_tensor().get_name()] = CPUTensorRole::OUTPUT;

        auto res = std::dynamic_pointer_cast<ngraph::op::Result>(op);
        if (!res->needs_copy())
        {
            shared_ptr<descriptor::TensorView> itv =
                res->get_inputs().at(0).get_output().get_tensor_view();
            function_output_index.emplace_back(tensor_data[itv->get_tensor().get_name()], i);
            m_tensor_roles[itv->get_tensor().get_name()] = CPUTensorRole::OUTPUT;
            tensor_alias[itv->get_tensor().get_name()] = tv->get_tensor().get_name();
            propagate_in_place_output(
                &(res->get_inputs().at(0).get_output()), tv->get_tensor().get_name(), true);
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
        auto handler = build_dispatcher.find(type_index(typeid(n)));
        if (handler == build_dispatcher.end())
        {
            throw ngraph_error("Unhandled op during executor construction : " +
                               node->description());
        }
        vector<TensorViewWrapper> in;
        vector<string> in_names;
        for (const descriptor::Input& input : node->get_inputs())
        {
            const descriptor::Output& output = input.get_output();
            shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
            in.push_back(TensorViewWrapper(tv, tv->get_tensor().get_name()));
            in_names.push_back(tv->get_tensor().get_name());
        }
        vector<TensorViewWrapper> out;
        vector<string> out_names;
        for (const descriptor::Output& output : node->get_outputs())
        {
            shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
            out.push_back(TensorViewWrapper(tv, tv->get_tensor().get_name()));
            out_names.push_back(tv->get_tensor().get_name());
        }

        m_op_attrs.emplace_back(node->description(), out_names, in_names);

        size_t functor_count = functors.size();
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

        enables.emplace_back(make_pair(enable, functors.size() - functor_count));
        enable_nodename_list.emplace_back(make_pair(enable, node->get_name()));
    }

    executor = [&](CPURuntimeContext* ctx, vector<void*>& inputs, vector<void*>& outputs) {
        cpu::Timestamp start_ts;
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
                std::unordered_map<
                    std::string,
                    tbb::flow::continue_node<tbb::flow::continue_msg, tbb::flow::lightweight>*>
                    nodename_tbbnode_map;
                tbb::flow::continue_node<tbb::flow::continue_msg,
                                         tbb::flow::lightweight>* flowgraph_node_start =
                    new tbb::flow::continue_node<tbb::flow::continue_msg, tbb::flow::lightweight>(
                        *(ctx->G), [&](const tbb::flow::continue_msg& msg) {});
                auto it = enable_nodename_list.begin();
                for (const auto& p : enables)
                {
                    std::vector<std::function<void(CPURuntimeContext*)>> ftrs;
                    for (size_t j = 0; j < p.second; j++)
                    {
                        ftrs.push_back(*functor);
                        std::advance(functor, 1);
                    }

                    tbb::flow::continue_node<tbb::flow::continue_msg, tbb::flow::lightweight>*
                        flowgraph_node = new tbb::flow::continue_node<tbb::flow::continue_msg,
                                                                      tbb::flow::lightweight>(
                            *(ctx->G), [&, ftrs](const tbb::flow::continue_msg& msg) {
                                if (p.first(ctx) || ctx->first_iteration)
                                {
                                    for (size_t j = 0; j < p.second; j++)
                                    {
                                        if (runtime::cpu::IsTracingEnabled())
                                        {
                                            start_ts = cpu::Clock::now();
                                        }
                                        ftrs[j](ctx);
                                        if (runtime::cpu::IsTracingEnabled())
                                        {
                                            ctx->op_durations[profiler_count++] =
                                                (std::chrono::duration_cast<cpu::Timescale>(
                                                     cpu::Clock::now() - start_ts))
                                                    .count();
                                        }
                                    }
                                }
                                else
                                {
                                    if (runtime::cpu::IsTracingEnabled())
                                    {
                                        for (size_t j = 0; j < p.second; j++)
                                        {
                                            ctx->op_durations[profiler_count++] = 0;
                                        }
                                    }
                                }
                            });
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
            (static_cast<
                 tbb::flow::continue_node<tbb::flow::continue_msg, tbb::flow::lightweight>*>(
                 &(*(ctx->G->begin()))))
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
            for (const auto& p : enables)
            {
                if (p.first(ctx) || ctx->first_iteration)
                {
                    for (size_t j = 0; j < p.second; j++)
                    {
                        if (runtime::cpu::IsTracingEnabled())
                        {
                            start_ts = cpu::Clock::now();
                        }
                        (*functor)(ctx);
                        if (runtime::cpu::IsTracingEnabled())
                        {
                            ctx->op_durations[profiler_count++] =
                                (std::chrono::duration_cast<cpu::Timescale>(cpu::Clock::now() -
                                                                            start_ts))
                                    .count();
                        }

                        std::advance(functor, 1);
                    }
                }
                else
                {
                    if (runtime::cpu::IsTracingEnabled())
                    {
                        for (size_t j = 0; j < p.second; j++)
                        {
                            ctx->op_durations[profiler_count++] = 0;
                        }
                    }
                    std::advance(functor, p.second);
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
        throw ngraph_error("Unhandled op during function emit : " + node.description());
    }
    vector<TensorViewWrapper> in;
    size_t arg_index = 0;
    set<string> arg_names;
    for (const descriptor::Input& input : node.get_inputs())
    {
        const descriptor::Output& output = input.get_output();
        shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
        TensorViewWrapper tvw{tv, "_arg" + to_string(arg_index)};
        if (!contains(arg_names, tvw.get_name()))
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
        shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
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
