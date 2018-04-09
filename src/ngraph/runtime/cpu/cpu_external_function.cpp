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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
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
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/result_copy_elimination.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/runtime/cpu/pass/cpu_layout.hpp"
#include "ngraph/runtime/cpu/pass/cpu_nop_elimination.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/op/allreduce.hpp"
#endif

using namespace std;
using namespace ngraph;

static const string s_output_dir = "cpu_codegen";

// Temporary Memory Pool alignment
static const size_t s_memory_pool_alignment = 4096;

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
    {TI(ngraph::op::ConvolutionBias), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionBias>},
    {TI(ngraph::op::ConvolutionRelu), &runtime::cpu::CPU_Emitter::emit<op::ConvolutionRelu>},
    // conv+bias backprop for data share the same implementation as ConvolutionBackpropData
    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
     &runtime::cpu::CPU_Emitter::emit<op::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::runtime::cpu::op::ConvertLayout),
     &runtime::cpu::CPU_Emitter::emit<runtime::cpu::op::ConvertLayout>},
    {TI(ngraph::op::Not), &runtime::cpu::CPU_Emitter::emit<op::Not>},
    {TI(ngraph::op::MaxPool), &runtime::cpu::CPU_Emitter::emit<op::MaxPool>},
    {TI(ngraph::op::Reverse), &runtime::cpu::CPU_Emitter::emit<op::Reverse>},
    {TI(ngraph::op::Result), &runtime::cpu::CPU_Emitter::emit<op::Result>},
    {TI(ngraph::op::ReduceWindow), &runtime::cpu::CPU_Emitter::emit<op::ReduceWindow>},
    {TI(ngraph::op::SelectAndScatter), &runtime::cpu::CPU_Emitter::emit<op::SelectAndScatter>},
    {TI(ngraph::op::AvgPool), &runtime::cpu::CPU_Emitter::emit<op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop), &runtime::cpu::CPU_Emitter::emit<op::AvgPoolBackprop>},
    {TI(ngraph::op::Pad), &runtime::cpu::CPU_Emitter::emit<op::Pad>},
    {TI(ngraph::op::BatchNorm), &runtime::cpu::CPU_Emitter::emit<op::BatchNorm>},
    {TI(ngraph::op::BatchNormRelu), &runtime::cpu::CPU_Emitter::emit<op::BatchNormRelu>},
    {TI(ngraph::op::BatchNormBackprop), &runtime::cpu::CPU_Emitter::emit<op::BatchNormBackprop>},
    {TI(ngraph::op::MaxPoolBackprop), &runtime::cpu::CPU_Emitter::emit<op::MaxPoolBackprop>},
    {TI(ngraph::op::Product), &runtime::cpu::CPU_Emitter::emit<op::Product>},
    {TI(ngraph::op::Max), &runtime::cpu::CPU_Emitter::emit<op::Max>},
    {TI(ngraph::op::Min), &runtime::cpu::CPU_Emitter::emit<op::Min>},
    {TI(ngraph::op::Relu), &runtime::cpu::CPU_Emitter::emit<op::Relu>},
    {TI(ngraph::op::ReluBackprop), &runtime::cpu::CPU_Emitter::emit<op::ReluBackprop>},
    {TI(ngraph::op::Sigmoid), &runtime::cpu::CPU_Emitter::emit<op::Sigmoid>},
    {TI(ngraph::op::Softmax), &runtime::cpu::CPU_Emitter::emit<op::Softmax>},
    {TI(ngraph::op::SigmoidBackprop), &runtime::cpu::CPU_Emitter::emit<op::SigmoidBackprop>},
};

runtime::cpu::CPU_ExternalFunction::CPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function, bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
    , m_compiled_function(nullptr)
    , m_emit_timing(false)
    , m_use_tbb(std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
    , m_function_name(function->get_name())
{
}

runtime::cpu::CPU_ExternalFunction::~CPU_ExternalFunction()
{
}

void runtime::cpu::CPU_ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    m_emit_timing = m_timing | (std::getenv("NGRAPH_CPU_EMIT_TIMING") != nullptr);

    m_mkldnn_emitter.reset(new MKLDNNEmitter());

    ngraph::pass::Manager pass_manager;

    pass_manager.register_pass<runtime::cpu::pass::CPUNopElimination>();
    pass_manager.register_pass<ngraph::pass::CoreFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUFusion>();
    pass_manager.register_pass<runtime::cpu::pass::CPUAssignment>(this);
    pass_manager.register_pass<runtime::cpu::pass::CPULayout>(this);
    pass_manager.register_pass<ngraph::pass::ResultCopyElimination>();
    pass_manager.register_pass<ngraph::pass::GetOutputElementElimination>();
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(s_memory_pool_alignment);
    pass_manager.run_passes(m_function);

    unordered_map<shared_ptr<Function>, list<shared_ptr<Node>>> function_ordered_ops;
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        function_ordered_ops.insert({current_function, current_function->get_ordered_ops()});
    }

    codegen::CodeWriter writer;

    writer +=
        R"(// Generated by the nGraph CPU backend
#include <cmath>
#include "ngraph/except.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_eigen_utils.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/reduce.hpp"
#include "ngraph/runtime/reference/reduce_window.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
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

    if (m_use_tbb)
    {
        writer << "#include <tbb/flow_graph.h>\n";
    }

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

    // This for loop creates a collection of functions that are called more than once
    // and emitting them as globally callable functions.
    // ops implement the is_functionally_identical method
    unordered_map<Node*, string> match_functions;
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        list<shared_ptr<Node>> tmp = function_ordered_ops.at(current_function);
        if (tmp.size() < 2)
        {
            // Since we are comparing ops there must be at least two ops to proceed.
            continue;
        }
        vector<shared_ptr<Node>> op_list{tmp.begin(), tmp.end()};
        unordered_map<const Node*, string> node_cache;
        for (size_t i = 0; i < op_list.size(); i++)
        {
            if (op_list[i]->is_constant() || op_list[i]->is_parameter())
            {
                continue;
            }

            Node& node = *op_list[i];
            auto handler = dispatcher.find(type_index(typeid(node)));
            if (handler == dispatcher.end())
            {
                throw ngraph_error("Unhandled op during code generation : " + node.description());
            }

            string s = emit_op_as_function(node, "f");
            node_cache.insert({&node, s});
        }
        for (size_t i = 0; i < op_list.size() - 1; i++)
        {
            if (op_list[i]->is_constant() || op_list[i]->is_parameter())
            {
                continue;
            }
            if (contains_key(match_functions, op_list[i].get()))
            {
                continue;
            }
            string match_function_name;
            for (size_t j = i + 1; j < op_list.size(); j++)
            {
                Node* op1 = op_list[i].get();
                Node* op2 = op_list[j].get();
                if (is_functionally_identical(*op1, *op2, node_cache))
                {
                    if (match_function_name.empty())
                    {
                        match_function_name = "func_" + op1->get_name();
                        match_functions.insert({op1, match_function_name});
                    }
                    match_functions.insert({op2, match_function_name});
                }
            }
            if (!match_function_name.empty())
            {
                writer << emit_op_as_function(*op_list[i], match_function_name);
            }
        }
    }

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

        writer << "extern \"C\" void " << current_function->get_name();
        writer << "(void** inputs, void** outputs, cpu::CPURuntimeContext* ctx)\n";
        writer << "{\n";
        writer.indent++;

        if (m_use_tbb)
        {
            // TODO: This should be static but we don't codegen statics correctly yet
            writer << "tbb::flow::graph G;\n\n";
        }

        // Execution tracing support
        if (runtime::cpu::IsTracingEnabled() && current_function->get_name() == m_function_name)
        {
            writer << "cpu::Timestamp start_ts;\n"
                   << "int profiler_count = 0;\n\n";
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
            size_t temp_pool_size = current_function->get_temporary_pool_size();
            writer << "// Allocate the memory pool\n";
            writer << "// Memory pool size is " << temp_pool_size << " bytes\n";
            writer << "// Worst case size is " << worst_case_tmp_size << " bytes\n";
            writer << "ngraph::runtime::AlignedBuffer memory_handler(" << temp_pool_size << ", "
                   << s_memory_pool_alignment << ");\n";
            writer << "size_t pool_base_ptr = (size_t)memory_handler.get_ptr();\n";
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
                }
            }
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
                arg_index++;
            }
        }

        // create output alias map
        /*
        size_t output_index = 0;
        unordered_map<descriptor::TensorView*, vector<size_t>> output_alias_map;
        vector<size_t> aliases;
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::TensorView> otv = op->get_output_tensor_view();
            vector<size_t>& al = output_alias_map[otv.get()];
            al.push_back(output_index);
            if (al.size() > 1)
            {
                aliases.push_back(output_index);
            }
            output_index++;
        }
        */

        // Add outputs to the variable name map
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            string type = tv->get_tensor_view_type()->get_element_type().c_type_string();
            stringstream ss;
            ss << "((" << type << "*)(outputs[" << i << "]))";
            m_variable_name_map[tv->get_tensor().get_name()] = ss.str();

            //it should be safe to assign both descriptors to one output*
            //since needs_copy == false makes `op::Result` an nop
            auto res = std::dynamic_pointer_cast<ngraph::op::Result>(op);
            if (!res->needs_copy())
            {
                shared_ptr<descriptor::TensorView> itv =
                    res->get_inputs().at(0).get_output().get_tensor_view();
                m_variable_name_map[itv->get_tensor().get_name()] = ss.str();
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
                    writer << "tbb::flow::continue_node<tbb::flow::continue_msg> "
                              "flowgraph_node_"
                           << node->get_name()
                           << "(G, [&](const tbb::flow::continue_msg &msg)\n{\n";
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
            string func_name;
            auto it = match_functions.find(node.get());
            if (it == match_functions.end())
            {
                handler->second(this, writer, node.get(), in, out);
            }
            else
            {
                func_name = it->second;
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

            //skip multi-output nodes since they would be covered by GetOutputElement
            if (node->get_output_size() == 1 &&
                //skip non-FP nodes
                (node->get_element_type() == element::f32 ||
                 node->get_element_type() == element::f64))
            {
                //check inputs and constants?
                if ((!node->is_parameter() && !node->is_constant()) ||
                    std::getenv("NGRAPH_CPU_CHECK_PARMS_AND_CONSTS"))
                {
                    if (std::getenv("NGRAPH_CPU_NAN_CHECK"))
                    {
                        generate_isnan_isinf_check(writer, node, out, "std::isnan");
                    }

                    if (std::getenv("NGRAPH_CPU_INF_CHECK"))
                    {
                        generate_isnan_isinf_check(writer, node, out, "std::isinf");
                    }
                }
            }

            // Emit operation epilogue
            if (!node->is_parameter() && !node->is_constant())
            {
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
            vector<Node*> dependence_graph_heads;

            traverse_nodes(
                current_function, [&writer, &dependence_graph_heads](shared_ptr<Node> n) {
                    if (!n->is_parameter() && !n->is_constant())
                    {
                        bool is_head = true;
                        for (auto arg : n->get_input_ops())
                        {
                            if (!arg->is_parameter() && !arg->is_constant())
                            {
                                is_head = false;
                                writer << "tbb::flow::make_edge(flowgraph_node_" << arg->get_name()
                                       << ", flowgraph_node_" << n->get_name() << ");\n";
                            }
                        }
                        if (is_head)
                        {
                            dependence_graph_heads.emplace_back(n.get());
                        }
                    }
                });

            writer << "\n";

            // Execute the flow graph
            if (!dependence_graph_heads.empty())
            {
                for (Node* n : dependence_graph_heads)
                {
                    writer << "flowgraph_node_" << n->get_name()
                           << ".try_put(tbb::flow::continue_msg());\n";
                }
                writer << "try { G.wait_for_all(); } catch(...) { throw; }\n";
            }
        }

        writer.indent--;
        // End generated function
        writer += "}\n\n";
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

    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

shared_ptr<ngraph::runtime::CallFrame> runtime::cpu::CPU_ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
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
