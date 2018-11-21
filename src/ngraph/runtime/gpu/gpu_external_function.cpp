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
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <fstream>
#include <mutex>
#include <string>
#include <tuple>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
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
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/common_function_collection.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"
#include "ngraph/runtime/gpu/op/batch_norm.hpp"
#include "ngraph/runtime/gpu/op/rnn.hpp"
#include "ngraph/runtime/gpu/pass/gpu_batch_norm_cache.hpp"
#include "ngraph/runtime/gpu/pass/gpu_layout.hpp"
#include "ngraph/runtime/gpu/pass/gpu_rnn_fusion.hpp"
#include "ngraph/runtime/gpu/pass/tensor_memory_reservation.hpp"

using namespace std;
using namespace ngraph;

static const string s_output_dir = "gpu_codegen";
static std::mutex s_compilation;

class GPUStaticInitializers
{
public:
    GPUStaticInitializers()
    {
        file_util::remove_directory(s_output_dir);
        file_util::make_directory(s_output_dir);
    }
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

static GPUStaticInitializers s_static_initializers;

void runtime::gpu::GPU_ExternalFunction::emit_op(GPU_ExternalFunction* external_function,
                                                 codegen::CodeWriter& writer,
                                                 const ngraph::Node* node,
                                                 const std::vector<GPUTensorWrapper>& args,
                                                 const std::vector<GPUTensorWrapper>& out)
{
    auto emit_function = GPU_Emitter::get_emit_function(*node);
    emit_function(external_function, writer, node, args, out);
};

const size_t runtime::gpu::GPU_ExternalFunction::GPU_ExternalFunction::s_memory_pool_alignment = 64;

runtime::gpu::GPU_ExternalFunction::GPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function,
    std::shared_ptr<GPU_Backend::BackendContext>& shared_context)
    : m_compiled_function(nullptr)
    , m_function(function)
    , m_emit_timing(false)
    , m_is_compiled(false)
    , m_shared_context(shared_context)
{
}

runtime::gpu::GPU_ExternalFunction::~GPU_ExternalFunction()
{
}

const string& runtime::gpu::GPU_ExternalFunction::get_pch_header_source()
{
    static string s_pch_header_source = R"(
// Generated by the nGraph GPU backend
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ngraph/runtime/gpu/cuda_error_check.hpp"
#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
)";
    return s_pch_header_source;
}

const string& runtime::gpu::GPU_ExternalFunction::get_header_source()
{
    static string s_header_source =
        get_pch_header_source() + R"(
using namespace ngraph;
using namespace ngraph::runtime;
using namespace std;
)"
        // The "dso_handle" symbol is required by __cxa_atexit()
        // which is enabled because the JIT uses it as the default mechanism
        // to register cleanup handlers. We use it, and not atexit(), because
        // atexit() happens too late, when the JIT is no longer alive
        + "void *__dso_handle = 0;\n\n" +
        "static gpu::GPURuntimeContext* m_runtime_context = nullptr;\n";

    return s_header_source;
}

void runtime::gpu::GPU_ExternalFunction::emit_header()
{
    m_writer << get_header_source();
}

void runtime::gpu::GPU_ExternalFunction::emit_timer_functions()
{
    if (m_emit_timing)
    {
        m_writer << "// Declare debug timers\n";
        vector<string> names;
        size_t index = 0;
        for (const auto& p : m_function_ordered_ops)
        {
            for (shared_ptr<Node> node : p.second)
            {
                if (!node->is_parameter() && !node->is_constant())
                {
                    names.push_back(node->get_name());
                    m_name_index_map.insert({node->get_name(), index++});
                }
            }
        }

        if (m_shared_context->m_runtime_context->stopwatch_pool == nullptr)
        {
            m_shared_context->m_runtime_context->stopwatch_pool = new StopWatchPool;
        }
        m_offset = m_shared_context->m_runtime_context->stopwatch_pool->size();
        m_shared_context->m_runtime_context->stopwatch_pool->allocate(names.size());
        m_writer << "extern \"C\" size_t get_debug_timer_count() { return " << names.size()
                 << "; }\n";
        m_writer << "extern \"C\" const char* get_debug_timer_name(size_t index)\n";
        m_writer.block_begin();
        m_writer << "static const char* timer_names[" << names.size() << "] =\n";
        m_writer.block_begin();
        vector<string> quoted_names;
        for (const string& name : names)
        {
            quoted_names.push_back("\"" + name + "\"");
        }
        m_writer << emit_string_array(quoted_names, 100 - (4 * 2 + 1));
        m_writer.indent--;
        m_writer << "\n};\n";
        m_writer << "return timer_names[index];\n";
        m_writer.block_end();

        m_writer << "extern \"C\" const size_t get_debug_timer_microseconds(size_t index)\n";
        m_writer.block_begin();
        m_writer << "return (index < " << names.size()
                 << " ? runtime::gpu::us_stopwatch(m_runtime_context, index + " << m_offset
                 << ") : 0);\n";
        m_writer.block_end();

        m_writer << "extern \"C\" const size_t get_debug_timer_call_count(size_t index)\n";
        m_writer.block_begin();
        m_writer << "return (index < " << names.size()
                 << " ? runtime::gpu::count_stopwatch(m_runtime_context, index + " << m_offset
                 << ") : 0);\n";
        m_writer.block_end();
        m_writer << "\n";
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_constant_declarations()
{
    m_writer << "// Declare all constants\n";
    for (const auto& p : m_function_ordered_ops)
    {
        for (shared_ptr<Node> node : p.second)
        {
            const op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
            if (c)
            {
                shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                // get an allocator for transient per kernel gpu memory
                runtime::gpu::GPUAllocator allocator =
                    m_shared_context->m_primitive_emitter->get_memory_allocator();
                size_t idx = allocator.reserve_argspace(c->get_data_ptr(),
                                                        tv->size() * tv->get_element_type().size());
                m_writer << "static size_t " << tv->get_name() << "_idx = " << idx << ";\n";
                m_writer << "static " << tv->get_element_type().c_type_string() << "* "
                         << tv->get_name() << " = nullptr;\n";
                m_variable_name_map[tv->get_name()] = tv->get_name();
            }
        }
    }

    m_writer << "\nstatic bool is_constant_mem_ptr_null = true;\n\n";
    m_writer << "static void invoke_constant_mem_ptr()\n";
    m_writer.block_begin();
    {
        m_writer << "if(is_constant_mem_ptr_null)\n";
        m_writer.block_begin();
        {
            for (const auto& p : m_function_ordered_ops)
            {
                for (shared_ptr<Node> node : p.second)
                {
                    const op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
                    if (c)
                    {
                        shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                        m_writer << tv->get_name() << " = reinterpret_cast<"
                                 << tv->get_element_type().c_type_string()
                                 << "*>(runtime::gpu::invoke_memory_primitive(m_runtime_context, "
                                 << tv->get_name() << "_idx));\n";
                    }
                }
            }
            m_writer << "is_constant_mem_ptr_null = false;\n";
        }
        m_writer.block_end();
    }
    m_writer.block_end();
}

void runtime::gpu::GPU_ExternalFunction::emit_function_declarations()
{
    m_writer << "// Declare all functions\n";
    for (const auto& p : m_function_ordered_ops)
    {
        m_writer << "extern \"C\" void " << p.first->get_name() << "(void** inputs, "
                 << "void** outputs, gpu::GPURuntimeContext* ctx);\n";
    }
    m_writer << "\n";
}

void runtime::gpu::GPU_ExternalFunction::emit_temp_mem_pool_allocation(
    shared_ptr<Function> current_function)
{
    bool temporaries_used = false;
    size_t worst_case_tmp_size = 0;
    for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
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
        m_writer << "// Allocate the memory pool\n";
        // TODO memory pool malloc.
        m_writer
            << "char* pool_base_ptr = (char*)ngraph::runtime::gpu::invoke_memory_primitive(ctx, "
            << m_tensor_memory_buffers.at(current_function->get_name()) << ");\n";

        // Add temporaries to the variable name map
        for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
        {
            for (descriptor::Tensor* tensor : node->liveness_new_list)
            {
                stringstream ss;
                ss << "((" << tensor->get_element_type().c_type_string() << "*)(pool_base_ptr + "
                   << tensor->get_pool_offset() << "))";
                m_variable_name_map[tensor->get_name()] = ss.str();
            }
        }
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_functions()
{
    for (const auto& p : m_function_ordered_ops)
    {
        auto current_function = p.first;
        set<string> output_names;
        for (shared_ptr<Node> op : current_function->get_results())
        {
            shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
            output_names.insert(tv->get_name());
        }
        set<descriptor::Tensor*> constants;
        for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
        {
            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                constants.insert(tv.get());
            }
        }

        m_writer << "extern \"C\" void " << current_function->get_name();
        m_writer << "(void** inputs, void** outputs, "
                 << "gpu::GPURuntimeContext* ctx) __attribute__ ((optnone))\n";
        m_writer.block_begin();
        {
            m_writer << "m_runtime_context = ctx;\n";
            // set constant pointers during the first run
            m_writer << "invoke_constant_mem_ptr();\n";

            // alocate temp memory pool
            emit_temp_mem_pool_allocation(current_function);

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
                    propagate_in_place_input(&param->get_outputs().at(i), ss.str());
                    arg_index++;
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

                auto res = dynamic_pointer_cast<ngraph::op::Result>(op);
                //keep assigning different outputs to a result descriptor
                //op::Result emitter will check if in and out descriptors are the same
                //and skip a copy
                auto input_node = res->get_inputs().at(0).get_output().get_node();
                if (!input_node->is_constant() && !input_node->is_parameter())
                {
                    shared_ptr<descriptor::Tensor> itv =
                        res->get_inputs().at(0).get_output().get_tensor_ptr();
                    auto output_name = ss.str();
                    m_variable_name_map[itv->get_name()] = output_name;
                    propagate_in_place_output(&(res->get_inputs().at(0).get_output()), output_name);
                }
            }

            for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
            {
                vector<GPUTensorWrapper> in;
                vector<string> node_input_names;
                vector<string> node_output_names;
                for (const descriptor::Input& input : node->get_inputs())
                {
                    const descriptor::Output& output = input.get_output();
                    shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                    in.push_back(GPUTensorWrapper(tv, m_variable_name_map[tv->get_name()]));
                    node_input_names.emplace_back(tv->get_name());
                }
                vector<GPUTensorWrapper> out;
                for (const descriptor::Output& output : node->get_outputs())
                {
                    shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                    out.push_back(GPUTensorWrapper(tv, m_variable_name_map[tv->get_name()]));
                    node_output_names.emplace_back(tv->get_name());
                }

                // Emit function description comment
                if (!node->is_parameter() && !node->is_constant())
                {
                    m_writer << "\n// " << node->get_name() << "(";
                    vector<string> parameter_nodes = node_input_names;
                    parameter_nodes.insert(
                        parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
                    m_writer << join(parameter_nodes);
                    m_writer << ")\n";
                    emit_debug_function_entry(node.get());
                }

                // Emit operation body
                auto it = m_node_function_map.find(node.get());
                if (it == m_node_function_map.end())
                {
                    emit_op(this, m_writer, node.get(), in, out);
                }
                else
                {
                    string func_name =
                        ngraph::pass::CommonFunctionCollection::create_function_name(*it->second);
                    vector<string> names;
                    for (const GPUTensorWrapper& tv : in)
                    {
                        names.push_back(tv.get_name());
                    }
                    for (const GPUTensorWrapper& tv : out)
                    {
                        names.push_back(tv.get_name());
                    }
                    names.push_back("ctx");
                    m_writer << func_name << "(" << join(names) << ");\n";
                }

                // Emit operation epilogue
                if (!node->is_parameter() && !node->is_constant())
                {
                    emit_debug_function_exit(node.get());
                }
            }
        }
        m_writer.block_end(); // End generated function
    }
}

void runtime::gpu::GPU_ExternalFunction::store_emitted_functions(const string& code)
{
    // TODO: Cleanup and make this a utility function
    string filename = file_util::path_join(s_output_dir, m_function_name + "_codegen.cpp");
    ofstream out(filename);
    out << code;
    out.close();
}

void runtime::gpu::GPU_ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }
    std::unique_lock<std::mutex> lock(s_compilation);

    m_function_name = m_function->get_name();

    auto allocator = std::make_shared<runtime::gpu::GPUAllocator>(
        m_shared_context->m_primitive_emitter->get_memory_allocator());

    ngraph::pass::Manager pass_manager;
#if CUDNN_VERSION >= 7200
    // recurrent network fusion
    pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::gpu::pass::RNNFusion>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<runtime::gpu::pass::MultiLayerRNNFusion>();
#else
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
#endif
    pass_manager.register_pass<runtime::gpu::pass::BatchNormCache>();
    pass_manager.register_pass<ngraph::pass::LikeReplacement>();
    pass_manager.register_pass<runtime::gpu::pass::GPULayout>(this);
    pass_manager.register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(s_memory_pool_alignment);
    pass_manager.register_pass<runtime::gpu::pass::TensorMemoryReservation>(
        *allocator, m_tensor_memory_buffers);
    std::string common_function_string;
    auto femitter = bind(&ngraph::runtime::gpu::GPU_ExternalFunction::emit_op_as_function,
                         this,
                         placeholders::_1,
                         placeholders::_2);
    pass_manager.register_pass<ngraph::pass::CommonFunctionCollection>(
        femitter, m_node_function_map, common_function_string);
    string dump_filename = file_util::path_join(s_output_dir, m_function_name + "_ops.txt");
    pass_manager.register_pass<ngraph::pass::DumpSorted>(dump_filename);

    pass_manager.run_passes(m_function);

    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        m_function_ordered_ops.emplace(current_function, current_function->get_ordered_ops());
    }

    emit_header();
    emit_timer_functions();
    emit_constant_declarations();
    emit_function_declarations();
    m_writer << common_function_string << "\n";
    emit_functions();

    // allocate device buffers for primitive arguments and workspace
    allocator->close();
    m_shared_context->m_primitive_emitter->allocate_primitive_memory();

    string code = m_writer.get_code();
    store_emitted_functions(code);

    m_compiler.reset(new codegen::Compiler());
    m_execution_engine.reset(new codegen::ExecutionEngine());
    m_compiler->set_precompiled_header_source(get_pch_header_source());

    auto codegen_module = m_compiler->compile(code);
    if (codegen_module == nullptr)
    {
        throw runtime_error("Function failed to compile to bitcode");
    }

    m_execution_engine->add_module(codegen_module);
    m_execution_engine->finalize();

    m_compiled_function = m_execution_engine->find_function<EntryPoint_t>(m_function_name);
    if (!m_compiled_function)
    {
        throw runtime_error("Function failed to compile");
    }

    m_is_compiled = true;
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_entry(Node* node)
{
    if (m_emit_timing)
    {
        m_writer << "runtime::gpu::start_stopwatch(ctx, "
                 << m_name_index_map[node->get_name()] + m_offset << ");\n";
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_exit(Node* node)
{
    if (m_emit_timing)
    {
        m_writer << "runtime::gpu::stop_stopwatch(ctx, "
                 << m_name_index_map[node->get_name()] + m_offset << ");\n";
    }
}

string runtime::gpu::GPU_ExternalFunction::emit_op_as_function(const Node& node,
                                                               const string& function_name)
{
    codegen::CodeWriter writer;
    writer << "static void " << function_name << "(";
    writer.indent++;
    vector<GPUTensorWrapper> in;
    size_t arg_index = 0;
    set<string> arg_names;
    for (const descriptor::Input& input : node.get_inputs())
    {
        const descriptor::Output& output = input.get_output();
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        GPUTensorWrapper tvw{tv, "_arg" + to_string(arg_index)};
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
    vector<GPUTensorWrapper> out;
    for (const descriptor::Output& output : node.get_outputs())
    {
        shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
        GPUTensorWrapper tvw{tv, "_out" + to_string(arg_index)};
        if (arg_index++ > 0)
        {
            writer << ",";
        }
        writer << "\n";
        writer << tvw.get_type() << "* " << tvw.get_name();
        out.push_back(tvw);
    }
    writer << ",\ngpu::GPURuntimeContext* ctx";
    writer.indent--;
    writer << "\n)\n";
    codegen::CodeWriter tmp_writer;
    emit_op(this, tmp_writer, &node, in, out);
    string body = tmp_writer.get_code();
    if (body.size() > 0 && body[0] == '{')
    {
        // Body already surrounded by curly braces so don't add more
        writer << body;
    }
    else
    {
        writer.block_begin();
        writer << body;
        writer.block_end();
    }

    string rc = writer.get_code();
    if (function_name == "f")
    {
        rc = strip_comments(rc);
    }
    return rc;
}

string runtime::gpu::GPU_ExternalFunction::strip_comments(const string& s) const
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

void runtime::gpu::GPU_ExternalFunction::propagate_in_place_input(
    ngraph::descriptor::Output* output, std::string input_name)
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

                        m_variable_name_map[output_tensor.get_name()] = input_name;

                        NGRAPH_DEBUG << "GPU codegen: Forwarding " << input_name << " through "
                                     << output_tensor.get_name();
                        stack.push_back(&c_op->get_outputs().at(output_index));
                    }
                }
            }
        }
    }
}

void runtime::gpu::GPU_ExternalFunction::propagate_in_place_output(
    ngraph::descriptor::Output* res_src_output, std::string output_name)
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

                        m_variable_name_map[input_tensor.get_name()] = output_name;

                        it = &arg->get_inputs().at(input_index).get_output();
                        propagate_further = true;
                    }
                }
            }
        }
    } while (propagate_further);
}
