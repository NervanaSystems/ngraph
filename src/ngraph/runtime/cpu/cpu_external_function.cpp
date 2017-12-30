// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

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
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concatenate.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_emitter.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"

using namespace std;
using namespace ngraph;

static const string s_output_dir = "cpu_codegen";

class StaticInitializers
{
public:
    StaticInitializers() { ngraph::file_util::remove_directory(s_output_dir); }
};

static StaticInitializers s_static_initializers;

#define TI(x) type_index(typeid(x))

static const runtime::cpu::OpMap dispatcher{
    {TI(ngraph::op::Add), &runtime::cpu::CPU_Emitter::EmitAdd},
    {TI(ngraph::op::Dot), &runtime::cpu::CPU_Emitter::EmitDot},
    {TI(ngraph::op::Multiply), &runtime::cpu::CPU_Emitter::EmitMultiply},
    {TI(ngraph::op::Parameter), &runtime::cpu::CPU_Emitter::EmitNop},
    {TI(ngraph::op::Abs), &runtime::cpu::CPU_Emitter::EmitAbs},
    {TI(ngraph::op::Concat), &runtime::cpu::CPU_Emitter::EmitConcat},
    {TI(ngraph::op::Divide), &runtime::cpu::CPU_Emitter::EmitDivide},
    {TI(ngraph::op::Equal), &runtime::cpu::CPU_Emitter::EmitEqual},
    {TI(ngraph::op::Greater), &runtime::cpu::CPU_Emitter::EmitGreater},
    {TI(ngraph::op::GreaterEq), &runtime::cpu::CPU_Emitter::EmitGreaterEq},
    {TI(ngraph::op::Less), &runtime::cpu::CPU_Emitter::EmitLess},
    {TI(ngraph::op::LessEq), &runtime::cpu::CPU_Emitter::EmitLessEq},
    {TI(ngraph::op::Log), &runtime::cpu::CPU_Emitter::EmitLog},
    {TI(ngraph::op::Maximum), &runtime::cpu::CPU_Emitter::EmitMaximum},
    {TI(ngraph::op::Minimum), &runtime::cpu::CPU_Emitter::EmitMinimum},
    {TI(ngraph::op::Negative), &runtime::cpu::CPU_Emitter::EmitNegative},
    {TI(ngraph::op::NotEqual), &runtime::cpu::CPU_Emitter::EmitNotEqual},
    {TI(ngraph::op::Power), &runtime::cpu::CPU_Emitter::EmitPower},
    {TI(ngraph::op::Select), &runtime::cpu::CPU_Emitter::EmitSelect},
    {TI(ngraph::op::Subtract), &runtime::cpu::CPU_Emitter::EmitSubtract},
    {TI(ngraph::op::Broadcast), &runtime::cpu::CPU_Emitter::EmitBroadcast},
    {TI(ngraph::op::Convert), &runtime::cpu::CPU_Emitter::EmitConvert},
    {TI(ngraph::op::Constant), &runtime::cpu::CPU_Emitter::EmitConstant},
    {TI(ngraph::op::Reshape), &runtime::cpu::CPU_Emitter::EmitReshape},
    {TI(ngraph::op::FunctionCall), &runtime::cpu::CPU_Emitter::EmitFunctionCall},
    {TI(ngraph::op::Reduce), &runtime::cpu::CPU_Emitter::EmitReduce},
    {TI(ngraph::op::Sign), &runtime::cpu::CPU_Emitter::EmitSign},
    {TI(ngraph::op::Slice), &runtime::cpu::CPU_Emitter::EmitSlice},
    {TI(ngraph::op::Sum), &runtime::cpu::CPU_Emitter::EmitSum},
    {TI(ngraph::op::Exp), &runtime::cpu::CPU_Emitter::EmitExp},
    {TI(ngraph::op::Sin), &runtime::cpu::CPU_Emitter::EmitSin},
    {TI(ngraph::op::Sinh), &runtime::cpu::CPU_Emitter::EmitSinh},
    {TI(ngraph::op::Cos), &runtime::cpu::CPU_Emitter::EmitCos},
    {TI(ngraph::op::Cosh), &runtime::cpu::CPU_Emitter::EmitCosh},
    {TI(ngraph::op::Tan), &runtime::cpu::CPU_Emitter::EmitTan},
    {TI(ngraph::op::Tanh), &runtime::cpu::CPU_Emitter::EmitTanh},
    {TI(ngraph::op::Asin), &runtime::cpu::CPU_Emitter::EmitAsin},
    {TI(ngraph::op::Acos), &runtime::cpu::CPU_Emitter::EmitAcos},
    {TI(ngraph::op::Atan), &runtime::cpu::CPU_Emitter::EmitAtan},
    {TI(ngraph::op::ReplaceSlice), &runtime::cpu::CPU_Emitter::EmitReplaceSlice},
    {TI(ngraph::op::OneHot), &runtime::cpu::CPU_Emitter::EmitOneHot},
    {TI(ngraph::op::Floor), &runtime::cpu::CPU_Emitter::EmitFloor},
    {TI(ngraph::op::Ceiling), &runtime::cpu::CPU_Emitter::EmitCeiling},
    {TI(ngraph::op::Sqrt), &runtime::cpu::CPU_Emitter::EmitSqrt},
    {TI(ngraph::op::Convolution), &runtime::cpu::CPU_Emitter::EmitConvolution},
    {TI(ngraph::op::Not), &runtime::cpu::CPU_Emitter::EmitNot},
    {TI(ngraph::op::MaxPool), &runtime::cpu::CPU_Emitter::EmitMaxPool},
};

runtime::cpu::CPU_ExternalFunction::CPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function, bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
    , m_compiled_function(nullptr)
    , m_emit_timing(std::getenv("NGRAPH_CPU_EMIT_TIMING") != nullptr)
    , m_use_tbb(std::getenv("NGRAPH_CPU_USE_TBB") != nullptr)
{
}

void runtime::cpu::CPU_ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    string function_name = m_function->get_name();
    string dump_filename = file_util::path_join(s_output_dir, function_name + "_ops.txt");

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::TopologicalSort>();
    // For now, just make everyone row-major.
    pass_manager.register_pass<pass::AssignLayout<descriptor::layout::DenseTensorViewLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.register_pass<pass::MemoryLayout>(64);
    pass_manager.register_pass<pass::DumpSorted>(dump_filename);
    pass_manager.run_passes(m_function);

    CPU_Emitter emitter;
    codegen::CodeWriter& writer = emitter.get_code_writer();

    writer +=
        R"(// Generated by the NGraph CPU backend
#include <cmath>

#include <tbb/flow_graph.h>

#include <Eigen/Dense>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_eigen_utils.hpp"
#include "ngraph/runtime/cpu/cpu_kernels.hpp"
#include "ngraph/runtime/kernel/broadcast.hpp"
#include "ngraph/runtime/kernel/concat.hpp"
#include "ngraph/runtime/kernel/convolution.hpp"
#include "ngraph/runtime/kernel/dot.hpp"
#include "ngraph/runtime/kernel/max_pool.hpp"
#include "ngraph/runtime/kernel/not.hpp"
#include "ngraph/runtime/kernel/one_hot.hpp"
#include "ngraph/runtime/kernel/reduce.hpp"
#include "ngraph/runtime/kernel/replace_slice.hpp"
#include "ngraph/runtime/kernel/slice.hpp"
#include "ngraph/runtime/kernel/sum.hpp"
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
        for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
        {
            for (shared_ptr<Node> node : current_function->get_ordered_ops())
            {
                if (!node->is_parameter() && !node->is_constant())
                {
                    names.push_back(node->get_name());
                }
            }
        }
        for (const string& s : names)
        {
            writer << "ngraph::stopwatch timer_" << s << ";\n";
        }
        writer << "extern \"C\" size_t get_debug_timer_count() { return " << names.size()
               << "; }\n";
        writer << "extern \"C\" const char* get_debug_timer_name(size_t index)\n";
        writer << "{\n";
        writer.indent++;
        writer << "const char* rc;\n";
        writer << "switch(index)\n";
        writer << "{\n";
        for (size_t i = 0; i < names.size(); i++)
        {
            writer << "case " << i << ": rc = \"" << names[i] << "\"; break;\n";
        }
        writer << "default: rc = \"\";\n";
        writer << "}\n";
        writer << "return rc;\n";
        writer.indent--;
        writer << "}\n";
        writer << "extern \"C\" const size_t get_debug_timer_microseconds(size_t index)\n";
        writer << "{\n";
        writer.indent++;
        writer << "size_t rc;\n";
        writer << "switch(index)\n";
        writer << "{\n";
        for (size_t i = 0; i < names.size(); i++)
        {
            writer << "case " << i << ": rc = timer_" << names[i]
                   << ".get_total_microseconds(); break;\n";
        }
        writer << "default: rc = 0;\n";
        writer << "}\n";
        writer << "return rc;\n";
        writer.indent--;
        writer << "}\n";
        writer << "extern \"C\" const size_t get_debug_timer_call_count(size_t index)\n";
        writer << "{\n";
        writer.indent++;
        writer << "size_t rc;\n";
        writer << "switch(index)\n";
        writer << "{\n";
        for (size_t i = 0; i < names.size(); i++)
        {
            writer << "case " << i << ": rc = timer_" << names[i] << ".get_call_count(); break;\n";
        }
        writer << "default: rc = 0;\n";
        writer << "}\n";
        writer << "return rc;\n";
        writer.indent--;
        writer << "}\n";
        writer << "\n";
    }

    writer << "// Declare all constants\n";
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            const op::Constant* c = dynamic_cast<op::Constant*>(node.get());
            if (c)
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                auto c_value_strings = c->get_value_strings();
                writer << "static " << tv->get_tensor().get_element_type().c_type_string() << " "
                       << tv->get_tensor().get_name() << "[" << c_value_strings.size() << "] =\n";
                writer << "{\n";
                for (size_t i = 0; i < c_value_strings.size(); i++)
                {
                    writer << "    " << c_value_strings[i] << ",\n";
                }
                writer << "};\n\n";
            }
        }
    }

    writer << "// Declare all functions\n";
    for (shared_ptr<Function> f : pass_manager.get_state().get_functions())
    {
        writer << "extern \"C\" void " << f->get_name() << "(void** inputs, void** outputs);\n";
    }
    writer << "\n";

    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        set<string> output_names;
        for (shared_ptr<Node> op : current_function->get_results())
        {
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            output_names.insert(tv->get_tensor().get_name());
        }
        set<descriptor::TensorView*> constants;
        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            if (dynamic_cast<op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                constants.insert(tv.get());
            }
        }

        writer << "extern \"C\" void " << current_function->get_name();
        writer << "(void** inputs, void** outputs)\n";
        writer << "{\n";
        writer.indent++;

        if (m_use_tbb)
        {
            // TODO: This should be static but we don't codegen statics correctly yet
            writer << "tbb::flow::graph G;\n\n";
        }

        bool temporaries_used = false;
        size_t worst_case_tmp_size = 0;
        for (shared_ptr<Node> node : current_function->get_ordered_ops())
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
                   << ngraph::runtime::cpu::alignment << ");\n";
            writer << "size_t pool_base_ptr = (size_t)memory_handler.get_ptr();\n";
            writer << "\n";

            writer << "// Define temporary tensors\n";
            for (shared_ptr<Node> node : current_function->get_ordered_ops())
            {
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    writer << tensor->get_element_type().c_type_string() << "* "
                           << tensor->get_name() << " = ("
                           << tensor->get_element_type().c_type_string() << "*)(pool_base_ptr + "
                           << tensor->get_pool_offset() << ");\n";
                }
            }
            writer << "\n";
        }

        writer << "// Define inputs\n";
        size_t arg_index = 0;
        for (shared_ptr<op::Parameter> param : current_function->get_parameters())
        {
            for (size_t i = 0; i < param->get_output_size(); ++i)
            {
                shared_ptr<descriptor::TensorView> tv = param->get_output_tensor_view(i);
                const element::Type& et = tv->get_tensor_view_type()->get_element_type();
                string type = et.c_type_string();
                writer << type << "* " << tv->get_tensor().get_name() << " = (" << type
                       << "*)(inputs[" << arg_index << "]);\n";
                arg_index++;
            }
        }
        writer << "\n";

        writer << "// Define outputs\n";
        // create alias list
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

        output_index = 0;
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            const element::Type& et = tv->get_tensor_view_type()->get_element_type();
            bool parameter_as_output = false;
            for (shared_ptr<op::Parameter> param : current_function->get_parameters())
            {
                for (const descriptor::Output& pout : param->get_outputs())
                {
                    shared_ptr<descriptor::TensorView> ptv = pout.get_tensor_view();
                    if (tv == ptv)
                    {
                        parameter_as_output = true;
                        writer << "memcpy(static_cast<" << et.c_type_string() << "*>(outputs["
                               << output_index << "]), " << ptv->get_tensor().get_name() << ", "
                               << ptv->get_tensor().size() << ");\n";
                        break;
                    }
                }
            }
            if (!parameter_as_output && !contains(aliases, output_index))
            {
                if (contains(constants, tv.get()))
                {
                    writer << "memcpy(outputs[" << output_index << "], "
                           << tv->get_tensor().get_name() << ", " << tv->get_tensor().size()
                           << ");\n";
                }
                else
                {
                    string type = et.c_type_string();
                    writer << type << "* " << tv->get_tensor().get_name() << " = static_cast<"
                           << type << "*>(outputs[" << output_index << "]);\n";
                }
            }
            output_index++;
        }
        writer << "\n";

        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            auto& n = *node; // Work around a compiler warning (*node inside typeid may have effects
            // with shared pointers, which is fine here but clang doesn't like it.)
            auto handler = dispatcher.find(type_index(typeid(n)));
            if (handler == dispatcher.end())
            {
                throw ngraph_error("Unhandled op during code generation : " + node->description());
            }
            vector<TensorViewWrapper> in;
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                in.push_back(TensorViewWrapper(tv));
            }
            vector<TensorViewWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                out.push_back(TensorViewWrapper(tv));
            }

            // Emit operation prologue
            if (!node->is_parameter() && !node->is_constant())
            {
                if (m_use_tbb)
                {
                    writer << "tbb::flow::continue_node<tbb::flow::continue_msg> flowgraph_node_"
                           << node->get_name()
                           << "(G, [&](const tbb::flow::continue_msg &msg)\n{\n";
                    writer.indent++;
                }
                if (m_emit_timing)
                {
                    emit_debug_function_entry(writer, node.get(), in, out);
                }
            }

            // Emit operation body
            handler->second(&emitter, node.get(), in, out);

            // Emit operation epilogue
            if (!node->is_parameter() && !node->is_constant())
            {
                handle_output_alias(writer, *node, output_alias_map);
                if (m_emit_timing)
                {
                    emit_debug_function_exit(writer, node.get(), in, out);
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

    // TODO: Cleanup and make this a utility function

    file_util::make_directory(s_output_dir);
    string filename = file_util::path_join(s_output_dir, function_name + "_codegen.cpp");
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
    m_compiled_function = m_execution_engine->find_function<EntryPoint_t>(function_name);
    assert(m_compiled_function);

    m_is_compiled = true;
    if (m_release_function)
    {
        release_function();
    }
}

void runtime::cpu::CPU_ExternalFunction::handle_output_alias(
    codegen::CodeWriter& writer,
    const Node& node,
    const unordered_map<descriptor::TensorView*, vector<size_t>>& output_alias_map)
{
    for (const descriptor::Output& output : node.get_outputs())
    {
        shared_ptr<descriptor::TensorView> otv = output.get_tensor_view();
        auto it = output_alias_map.find(otv.get());
        if (it != output_alias_map.end())
        {
            const vector<size_t>& outputs = it->second;
            if (outputs.size() > 1)
            {
                writer << "{    // handle output alias for previous op\n";
                writer.indent++;
                for (size_t i = 1; i < outputs.size(); i++)
                {
                    writer << "memcpy(static_cast<void*>(outputs[" << outputs[i]
                           << "]), static_cast<void*>(outputs[" << outputs[0] << "]), "
                           << otv->get_tensor().size() << ");\n";
                }
                writer.indent--;
                writer << "}\n";
            }
        }
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

void runtime::cpu::CPU_ExternalFunction::emit_debug_function_entry(
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<TensorViewWrapper>& in,
    const std::vector<TensorViewWrapper>& out)
{
    writer << "timer_" << node->get_name() << ".start();\n";
}

void runtime::cpu::CPU_ExternalFunction::emit_debug_function_exit(
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<TensorViewWrapper>& in,
    const std::vector<TensorViewWrapper>& out)
{
    writer << "timer_" << node->get_name() << ".stop();\n";
}
