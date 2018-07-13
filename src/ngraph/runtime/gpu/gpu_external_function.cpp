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
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <fstream>
#include <string>
#include <tuple>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
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
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"

using namespace std;
using namespace ngraph;

static const string s_output_dir = "gpu_codegen";

class StaticInitializers
{
public:
    StaticInitializers()
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

static StaticInitializers s_static_initializers;

#define TI(x) type_index(typeid(x))

static const runtime::gpu::OpMap dispatcher{
    {TI(ngraph::op::Add), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Add>},
    {TI(ngraph::op::Dot), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Dot>},
    {TI(ngraph::op::Multiply), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Multiply>},
    {TI(ngraph::op::Parameter), &runtime::gpu::GPU_Emitter::nop},
    {TI(ngraph::op::Abs), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Abs>},
    {TI(ngraph::op::Concat), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Concat>},
    {TI(ngraph::op::Divide), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Divide>},
    {TI(ngraph::op::Equal), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Equal>},
    {TI(ngraph::op::GetOutputElement),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::GetOutputElement>},
    {TI(ngraph::op::Greater), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Greater>},
    {TI(ngraph::op::GreaterEq),
     &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::GreaterEq>},
    {TI(ngraph::op::Less), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Less>},
    {TI(ngraph::op::LessEq), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::LessEq>},
    {TI(ngraph::op::Log), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Log>},
    {TI(ngraph::op::Maximum), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Maximum>},
    {TI(ngraph::op::Minimum), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Minimum>},
    {TI(ngraph::op::Negative), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Negative>},
    {TI(ngraph::op::NotEqual), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::NotEqual>},
    {TI(ngraph::op::Power), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Power>},
    {TI(ngraph::op::Select), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Select>},
    {TI(ngraph::op::Subtract), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Subtract>},
    {TI(ngraph::op::Broadcast), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Broadcast>},
    {TI(ngraph::op::Convert), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Convert>},
    {TI(ngraph::op::Constant), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Constant>},
    {TI(ngraph::op::Reshape), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Reshape>},
    {TI(ngraph::op::FunctionCall), &runtime::gpu::GPU_Emitter::emit<ngraph::op::FunctionCall>},
    {TI(ngraph::op::Reduce), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Reduce>},
    {TI(ngraph::op::Sign), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Sign>},
    {TI(ngraph::op::Slice), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Slice>},
    {TI(ngraph::op::Sum), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Sum>},
    {TI(ngraph::op::Exp), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Exp>},
    {TI(ngraph::op::Sin), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Sin>},
    {TI(ngraph::op::Sinh), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Sinh>},
    {TI(ngraph::op::Cos), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Cos>},
    {TI(ngraph::op::Cosh), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Cosh>},
    {TI(ngraph::op::Tan), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Tan>},
    {TI(ngraph::op::Tanh), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Tanh>},
    {TI(ngraph::op::Asin), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Asin>},
    {TI(ngraph::op::Acos), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Acos>},
    {TI(ngraph::op::Atan), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Atan>},
    {TI(ngraph::op::ReplaceSlice), &runtime::gpu::GPU_Emitter::emit<ngraph::op::ReplaceSlice>},
    {TI(ngraph::op::OneHot), &runtime::gpu::GPU_Emitter::emit<ngraph::op::OneHot>},
    {TI(ngraph::op::Floor), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Floor>},
    {TI(ngraph::op::Ceiling), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Ceiling>},
    {TI(ngraph::op::Sqrt), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Sqrt>},
    {TI(ngraph::op::Convolution), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Convolution>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::ConvolutionBackpropData>},
    {TI(ngraph::op::Not), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Not>},
    {TI(ngraph::op::MaxPool), &runtime::gpu::GPU_Emitter::emit<ngraph::op::MaxPool>},
    {TI(ngraph::op::Reverse), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Reverse>},
    {TI(ngraph::op::ReverseSequence),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::ReverseSequence>},
    {TI(ngraph::op::Result), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Result>},
    {TI(ngraph::op::ReduceWindow), &runtime::gpu::GPU_Emitter::emit<ngraph::op::ReduceWindow>},
    {TI(ngraph::op::SelectAndScatter),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::SelectAndScatter>},
    {TI(ngraph::op::AvgPool), &runtime::gpu::GPU_Emitter::emit<ngraph::op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::AvgPoolBackprop>},
    {TI(ngraph::op::Pad), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Pad>},
    {TI(ngraph::op::BatchNorm), &runtime::gpu::GPU_Emitter::emit<ngraph::op::BatchNorm>},
    {TI(ngraph::op::BatchNormBackprop),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::BatchNormBackprop>},
    {TI(ngraph::op::MaxPoolBackprop),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::MaxPoolBackprop>},
    {TI(ngraph::op::Product), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Product>},
    {TI(ngraph::op::Max), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Max>},
    {TI(ngraph::op::Min), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Min>},
    {TI(ngraph::op::Relu), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Relu>},
    {TI(ngraph::op::ReluBackprop),
     &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::ReluBackprop>},
    {TI(ngraph::op::Softmax), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Softmax>},
    {TI(ngraph::op::And), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::And>},
    {TI(ngraph::op::Or), &runtime::gpu::GPU_Emitter::emit_elementwise<ngraph::op::Or>}};

runtime::gpu::GPU_ExternalFunction::GPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function, bool release_function)
    : m_compiled_function(nullptr)
    , m_ctx(new GPURuntimeContext)
    , m_function(function)
    , m_emit_timing(false)
    , m_is_compiled(false)
    , m_release_function(release_function)
    , m_temporaries_used(false)
{
    // Create context use driver API and make it current, the runtime call will pickup the context
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
    // #interoperability-between-runtime-and-driver-apis
    ngraph::runtime::gpu::CudaContextManager::Instance().SetContextCurrent();

    cublasStatus_t cublasStatus = cublasCreate(&m_cublas_handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        throw runtime_error("cuBLAS create handle failed");
    }
    cudnnStatus_t cudnnStatus = cudnnCreate(&m_cudnn_handle);
    if (cudnnStatus != CUDNN_STATUS_SUCCESS)
    {
        throw runtime_error("cuDNN create handle failed");
    }
    // Pass scalars as reference on the Device
    cublasSetPointerMode(m_cublas_handle, CUBLAS_POINTER_MODE_DEVICE);

    // register with c-api runtime context
    m_ctx->cublas_handle = &m_cublas_handle;
    m_ctx->cudnn_handle = &m_cudnn_handle;
    m_ctx->compiled_kernel_pool = new CudaFunctionPool;
}

runtime::gpu::GPU_ExternalFunction::~GPU_ExternalFunction()
{
    cublasDestroy(m_cublas_handle);
    cudnnDestroy(m_cudnn_handle);
    delete m_ctx->compiled_kernel_pool;
}

void runtime::gpu::GPU_ExternalFunction::emit_header()
{
    m_writer += R"(
// Generated by the nGraph GPU backend
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/gpu/cudnn_descriptors.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_ops.hpp"
#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/util.hpp"
)";

    m_pch_header_source = m_writer.get_code();

    m_writer += R"(
using namespace ngraph;
using namespace ngraph::runtime;
using namespace std;
)";

    // The "dso_handle" symbol is required by __cxa_atexit()
    // which is enabled because the JIT uses it as the default mechanism
    // to register cleanup handlers. We use it, and not atexit(), because
    // atexit() happens too late, when the JIT is no longer alive
    m_writer << "void *__dso_handle = 0;\n\n";
}

void runtime::gpu::GPU_ExternalFunction::emit_timer_functions()
{
    if (m_emit_timing)
    {
        m_writer << "// Declare debug timers\n";
        vector<string> names;
        size_t index = 0;
        for (shared_ptr<Function> current_function : m_pass_manager.get_state().get_functions())
        {
            for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
            {
                if (!node->is_parameter() && !node->is_constant())
                {
                    names.push_back(node->get_name());
                    m_name_index_map.insert({node->get_name(), index++});
                }
            }
        }
        writer << "runtime::gpu::stopwatch timers[" << names.size() << "];\n";
        writer << "extern \"C\" size_t get_debug_timer_count() { return " << names.size()
               << "; }\n";
        writer << "extern \"C\" const char* get_debug_timer_name(size_t index)\n";
        writer.block_begin();
        writer << "static const char* timer_names[" << names.size() << "] =\n";
        writer.block_begin();
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
                 << " ? timers[index].get_total_microseconds() : 0);\n";
        m_writer.block_end();

        m_writer << "extern \"C\" const size_t get_debug_timer_call_count(size_t index)\n";
        m_writer.block_begin();
        m_writer << "return (index < " << names.size()
                 << " ? timers[index].get_call_count() : 0);\n";
        m_writer.block_end();
        m_writer << "\n";
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_constant_declarations()
{
    m_writer << "// Declare all constants\n";
    for (shared_ptr<Function> current_function : m_pass_manager.get_state().get_functions())
    {
        for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
        {
            const op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
            if (c)
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                // get an allocator for transient per kernel gpu memory
                GPUAllocator allocator = this->m_primitive_emitter->get_memory_allocator();
                size_t idx = allocator.reserve_argspace(
                    c->get_data_ptr(),
                    tv->get_tensor().size() * tv->get_tensor().get_element_type().size());
                m_writer << "static size_t " << tv->get_tensor().get_name() << "_idx = " << idx
                         << ";\n";
                m_writer << "static " << tv->get_tensor().get_element_type().c_type_string() << "* "
                         << tv->get_tensor().get_name() << " = nullptr;\n";
                m_variable_name_map[tv->get_tensor().get_name()] = tv->get_tensor().get_name();
            }
        }
    }

    m_writer << "\nstatic bool is_constant_mem_ptr_null = true;\n\n";
    m_writer << "static void invoke_constant_mem_ptr(gpu::GPURuntimeContext* ctx)\n";
    m_writer.block_begin();
    {
        m_writer << "if(is_constant_mem_ptr_null)\n";
        m_writer.block_begin();
        {
            for (shared_ptr<Function> current_function : m_pass_manager.get_state().get_functions())
            {
                for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
                {
                    const op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
                    if (c)
                    {
                        shared_ptr<descriptor::TensorView> tv =
                            node->get_outputs()[0].get_tensor_view();
                        m_writer << tv->get_tensor().get_name() << " = reinterpret_cast<"
                                 << tv->get_tensor().get_element_type().c_type_string()
                                 << "*>(runtime::gpu::invoke_memory_primitive(ctx, "
                                 << tv->get_tensor().get_name() << "_idx));\n";
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
    for (shared_ptr<Function> f : m_pass_manager.get_state().get_functions())
    {
        m_writer << "extern \"C\" void " << f->get_name() << "(void** inputs, void** outputs, "
                 << "gpu::GPURuntimeContext* ctx);\n";
    }
    m_writer << "\n";
}

void runtime::gpu::GPU_ExternalFunction::collect_unique_functions()
{
    // This for loop creates a collection of functions that are called more than once
    // and emitting them as globally callable functions.
    // ops implement the is_functionally_identical method
    unordered_map<string, string> match_function_map;
    for (shared_ptr<Function> current_function : m_pass_manager.get_state().get_functions())
    {
        list<shared_ptr<Node>> tmp = m_function_ordered_ops.at(current_function);
        if (tmp.size() < 2)
        {
            // Since we are comparing ops there must be at least two ops to proceed.
            continue;
        }
        vector<shared_ptr<Node>> op_list{tmp.begin(), tmp.end()};
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

            string match_function = emit_op_as_function(node, "__f__");
            string match_function_name;
            if (contains_key(match_function_map, match_function))
            {
                match_function_name = match_function_map[match_function];
            }
            else
            {
                auto offset = match_function.find("__f__");
                string emitted_function = match_function;
                match_function_name = "func_" + node.get_name();
                emitted_function.replace(offset, 5, match_function_name);
                match_function_map.insert({match_function, match_function_name});
                m_writer << emitted_function << "\n";
            }
            m_node_function_map.insert({&node, match_function_name});
        }
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_temp_mem_pool_allocation(
    shared_ptr<Function> current_function)
{
    m_temporaries_used = false;
    size_t worst_case_tmp_size = 0;
    for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
    {
        if (node->liveness_new_list.size() > 0)
        {
            m_temporaries_used = true;
            for (descriptor::Tensor* tensor : node->liveness_new_list)
            {
                worst_case_tmp_size += tensor->size();
            }
        }
    }
    if (m_temporaries_used)
    {
        size_t temp_pool_size = current_function->get_temporary_pool_size();
        m_writer << "// Allocate the memory pool\n";
        // TODO memory pool malloc.
        m_writer << "void* pool_base_ptr = ngraph::runtime::gpu::create_gpu_buffer("
                 << temp_pool_size << ");\n";

        // Add temporaries to the variable name map
        for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
        {
            for (descriptor::Tensor* tensor : node->liveness_new_list)
            {
                stringstream ss;
                ss << "((" << tensor->get_element_type().c_type_string()
                   << "*)((char *)pool_base_ptr + " << tensor->get_pool_offset() << "))";
                m_variable_name_map[tensor->get_name()] = ss.str();
            }
        }
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_temp_mem_pool_release()
{
    if (m_temporaries_used)
    {
        m_writer << "ngraph::runtime::gpu::free_gpu_buffer(pool_base_ptr);\n";
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_functions()
{
    for (shared_ptr<Function> current_function : m_pass_manager.get_state().get_functions())
    {
        set<string> output_names;
        for (shared_ptr<Node> op : current_function->get_results())
        {
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            output_names.insert(tv->get_tensor().get_name());
        }
        set<descriptor::TensorView*> constants;
        for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
        {
            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                constants.insert(tv.get());
            }
        }

        m_writer << "extern \"C\" void " << current_function->get_name();
        m_writer << "(void** inputs, void** outputs, "
                 << "gpu::GPURuntimeContext* ctx)\n";
        m_writer.block_begin();
        {
            //set constant pointers during the first run
            m_writer << "invoke_constant_mem_ptr(ctx);\n";

            //alocate temp memory pool
            emit_temp_mem_pool_allocation(current_function);

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
                auto res = dynamic_pointer_cast<ngraph::op::Result>(op);
                if (!res->needs_copy())
                {
                    shared_ptr<descriptor::TensorView> itv =
                        res->get_inputs().at(0).get_output().get_tensor_view();
                    m_variable_name_map[itv->get_tensor().get_name()] = ss.str();
                }
            }

            for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
            {
                auto& n =
                    *node; // Work around a compiler warning (*node inside typeid may have effects
                // with shared pointers, which is fine here but clang doesn't like it.)
                auto handler = dispatcher.find(type_index(typeid(n)));
                if (handler == dispatcher.end())
                {
                    throw ngraph_error("Unhandled op during code generation : " +
                                       node->description());
                }
                vector<GPU_TensorViewWrapper> in;
                vector<string> node_input_names;
                vector<string> node_output_names;
                for (const descriptor::Input& input : node->get_inputs())
                {
                    const descriptor::Output& output = input.get_output();
                    shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                    in.push_back(GPU_TensorViewWrapper(
                        tv, m_variable_name_map[tv->get_tensor().get_name()]));
                    node_input_names.emplace_back(tv->get_tensor().get_name());
                }
                vector<GPU_TensorViewWrapper> out;
                for (const descriptor::Output& output : node->get_outputs())
                {
                    shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                    out.push_back(GPU_TensorViewWrapper(
                        tv, m_variable_name_map[tv->get_tensor().get_name()]));
                    node_output_names.emplace_back(tv->get_tensor().get_name());
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
                string func_name;
                func_name = m_node_function_map[node.get()];
                if (func_name.empty())
                {
                    //throw runtime_error("No matching function found for '" + node->get_name() + "'");
                    handler->second(this, m_writer, node.get(), in, out);
                }
                else
                {
                    vector<string> names;
                    for (const GPU_TensorViewWrapper& tv : in)
                    {
                        names.push_back(tv.get_name());
                    }
                    for (const GPU_TensorViewWrapper& tv : out)
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
            emit_temp_mem_pool_release();
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

    m_primitive_emitter.reset(new GPUPrimitiveEmitter());

    m_function_name = m_function->get_name();
    string dump_filename = file_util::path_join(s_output_dir, m_function_name + "_ops.txt");

    // For now, just make everyone row-major.
    m_pass_manager.register_pass<pass::ResultCopyElimination>();
    m_pass_manager.register_pass<pass::AssignLayout<descriptor::layout::DenseTensorViewLayout>>();
    m_pass_manager.register_pass<pass::Liveness>();
    m_pass_manager.register_pass<pass::MemoryLayout>(64);
    m_pass_manager.register_pass<pass::DumpSorted>(dump_filename);
    m_pass_manager.run_passes(m_function);

    for (shared_ptr<Function> current_function : m_pass_manager.get_state().get_functions())
    {
        m_function_ordered_ops.insert({current_function, current_function->get_ordered_ops()});
    }

    emit_header();
    emit_timer_functions();
    emit_constant_declarations();
    emit_function_declarations();
    collect_unique_functions();
    emit_functions();
    // allocate device buffers for primitive arguments and workspace
    m_primitive_emitter->allocate_primitive_memory();

    string code = m_writer.get_code();
    store_emitted_functions(code);

    m_compiler.reset(new codegen::Compiler());
    m_execution_engine.reset(new codegen::ExecutionEngine());
    m_compiler->set_precompiled_header_source(m_pch_header_source);

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
    if (m_release_function)
    {
        release_function();
    }
}

shared_ptr<ngraph::runtime::gpu::GPU_CallFrame>
    runtime::gpu::GPU_ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }

    return make_shared<GPU_CallFrame>(shared_from_this(), m_compiled_function);
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_entry(Node* node)
{
    if (m_emit_timing)
    {
        m_writer << "timers[" << m_name_index_map[node->get_name()] << "].start();\n";
    }
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_exit(Node* node)
{
    if (m_emit_timing)
    {
        m_writer << "timers[" << m_name_index_map[node->get_name()] << "].stop();\n";
    }
}

unique_ptr<runtime::gpu::GPURuntimeContext>& runtime::gpu::GPU_ExternalFunction::ctx()
{
    return m_ctx;
}

string runtime::gpu::GPU_ExternalFunction::emit_op_as_function(const Node& node,
                                                               const string& function_name)
{
    codegen::CodeWriter writer;
    writer << "static void " << function_name << "(";
    writer.indent++;
    // Work around a compiler warning (*node inside typeid may have effects
    // with shared pointers, which is fine here but clang doesn't like it.)
    auto handler = dispatcher.find(type_index(typeid(node)));
    vector<GPU_TensorViewWrapper> in;
    size_t arg_index = 0;
    set<string> arg_names;
    for (const descriptor::Input& input : node.get_inputs())
    {
        const descriptor::Output& output = input.get_output();
        shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
        GPU_TensorViewWrapper tvw{tv, "_arg" + to_string(arg_index)};
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
    vector<GPU_TensorViewWrapper> out;
    for (const descriptor::Output& output : node.get_outputs())
    {
        shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
        GPU_TensorViewWrapper tvw{tv, "_out" + to_string(arg_index)};
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
    handler->second(this, tmp_writer, &node, in, out);
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
