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
#include <cudnn_v7.h>
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
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/allreduce.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concat.hpp"
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
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/max.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/min.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/ops/pad.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/product.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/reduce_window.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/ops/remainder.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/result.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/select_and_scatter.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/softmax.hpp"
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
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"

using namespace std;
using namespace ngraph;

static const string s_output_dir = "gpu_codegen";

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

static const runtime::gpu::OpMap dispatcher{
    {TI(ngraph::op::Add), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Add>},
    {TI(ngraph::op::Dot), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Dot>},
    {TI(ngraph::op::Multiply), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Multiply>},
    {TI(ngraph::op::Parameter), &runtime::gpu::GPU_Emitter::nop},
    {TI(ngraph::op::Abs), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Concat), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Concat>},
    {TI(ngraph::op::Divide), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Equal), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::GetOutputElement),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::GetOutputElement>},
    {TI(ngraph::op::Greater), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::GreaterEq), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Less), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::LessEq), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Log), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Maximum), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Maximum>},
    {TI(ngraph::op::Minimum), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Minimum>},
    {TI(ngraph::op::Negative), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Negative>},
    {TI(ngraph::op::NotEqual), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Power), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Select), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Select>},
    {TI(ngraph::op::Subtract), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Broadcast), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Broadcast>},
    {TI(ngraph::op::Convert), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Constant), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Constant>},
    {TI(ngraph::op::Reshape), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Reshape>},
    {TI(ngraph::op::FunctionCall), &runtime::gpu::GPU_Emitter::emit<ngraph::op::FunctionCall>},
    {TI(ngraph::op::Reduce), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Reduce>},
    {TI(ngraph::op::Sign), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Slice), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Slice>},
    {TI(ngraph::op::Sum), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Sum>},
    {TI(ngraph::op::Exp), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Sin), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Sinh), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Cos), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Cosh), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Tan), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Tanh), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Asin), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Acos), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Atan), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::ReplaceSlice), &runtime::gpu::GPU_Emitter::emit<ngraph::op::ReplaceSlice>},
    {TI(ngraph::op::OneHot), &runtime::gpu::GPU_Emitter::emit<ngraph::op::OneHot>},
    {TI(ngraph::op::Floor), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Ceiling), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::Sqrt), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Sqrt>},
    {TI(ngraph::op::Convolution), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Convolution>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::gpu::GPU_Emitter::emit<ngraph::op::ConvolutionBackpropData>},
    {TI(ngraph::op::Not), &runtime::gpu::GPU_Emitter::EmitElementwise},
    {TI(ngraph::op::MaxPool), &runtime::gpu::GPU_Emitter::emit<ngraph::op::MaxPool>},
    {TI(ngraph::op::Reverse), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Reverse>},
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
    {TI(ngraph::op::Relu), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Relu>},
    {TI(ngraph::op::ReluBackprop), &runtime::gpu::GPU_Emitter::emit<ngraph::op::ReluBackprop>},
    {TI(ngraph::op::Softmax), &runtime::gpu::GPU_Emitter::emit<ngraph::op::Softmax>},
};

runtime::gpu::GPU_ExternalFunction::GPU_ExternalFunction(
    const shared_ptr<ngraph::Function>& function, bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
    , m_compiled_function(nullptr)
    , m_emit_timing(std::getenv("NGRAPH_GPU_EMIT_TIMING") != nullptr)
{
}

void runtime::gpu::GPU_ExternalFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }

    string function_name = m_function->get_name();
    string dump_filename = file_util::path_join(s_output_dir, function_name + "_ops.txt");

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::TopologicalSort>();
    // For now, just make everyone row-major.
    pass_manager.register_pass<pass::AssignLayout<descriptor::layout::DenseTensorViewLayout>>();
    pass_manager.register_pass<pass::Liveness>();
    pass_manager.register_pass<pass::MemoryLayout>(64);
    pass_manager.register_pass<pass::DumpSorted>(dump_filename);
    pass_manager.run_passes(m_function);

    codegen::CodeWriter writer;

    writer +=
        R"(// Generated by the NGraph GPU backend
    #include <cublas_v2.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cudnn_v7.h>

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
    #include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
    #include "ngraph/runtime/gpu/gpu_cuda_kernel_ops.hpp"
    #include "ngraph/runtime/gpu/gpu_util.hpp"
    #include "ngraph/util.hpp"
)";

    string pch_header_source = writer.get_code();

    writer += R"(
using namespace ngraph;
using namespace std;
    )";

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
    //     // The "dso_handle" symbol is required by __cxa_atexit()
    //     // which is enabled because the JIT uses it as the default mechanism
    //     // to register cleanup handlers. We use it, and not atexit(), because
    //     // atexit() happens too late, when the JIT is no longer alive

    writer << "void *__dso_handle = 0;\n\n";
    writer << "// Declare all constants\n";
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            const op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
            if (c)
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                auto c_value_strings = c->get_value_strings();
                writer << "static " << tv->get_tensor().get_element_type().c_type_string() << " "
                       << tv->get_tensor().get_name() << "_cpu[" << c_value_strings.size()
                       << "] =\n";
                writer << "{\n";
                writer.indent++;
                writer << emit_string_array(c_value_strings, 100 - writer.indent * 4);
                writer.indent--;
                writer << "\n};\n\n";
                writer << "static " << tv->get_tensor().get_element_type().c_type_string() << " *"
                       << tv->get_tensor().get_name() << ";\n";
                m_variable_name_map[tv->get_tensor().get_name()] = tv->get_tensor().get_name();
            }
        }
    }

    writer << "// Declare all functions\n";
    for (shared_ptr<Function> f : pass_manager.get_state().get_functions())
    {
        writer << "extern \"C\" void " << f->get_name() << "(void** inputs, void** outputs, "
                                                           "cublasHandle_t& cublas_handle, "
                                                           "cudnnHandle_t& cudnn_handle);\n";
    }

    writer << "\n";

    unordered_map<Node*, string> match_functions;
    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        set<string> output_names;
        for (shared_ptr<Node> op : current_function->get_results())
        {
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            output_names.insert(tv->get_tensor().get_name());
        }
        const list<shared_ptr<Node>>& tmp = current_function->get_ordered_ops();
        if (tmp.size() < 2)
        {
            // Since we are comparing ops there must be at least two ops to proceed.
            continue;
        }
        vector<shared_ptr<Node>> op_list{tmp.begin(), tmp.end()};
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
            if (!match_function_name.empty())
            {
                writer << "static void " << match_function_name << "(";
                writer.indent++;
                // Work around a compiler warning (*node inside typeid may have effects
                // with shared pointers, which is fine here but clang doesn't like it.)
                auto& n = *op_list[i];
                auto handler = dispatcher.find(type_index(typeid(n)));
                vector<GPU_TensorViewWrapper> in;
                size_t arg_index = 0;
                set<string> arg_names;
                for (const descriptor::Input& input : n.get_inputs())
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
                for (const descriptor::Output& output : n.get_outputs())
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
                writer.indent--;
                writer << "\n)\n";
                writer << "{\n";
                writer.indent++;
                handler->second(this, writer, &n, in, out);
                writer.indent--;
                writer << "}\n";
            }
        }
    }

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
            if (dynamic_cast<ngraph::op::Constant*>(node.get()))
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                constants.insert(tv.get());
            }
        }

        writer << "extern \"C\" void " << current_function->get_name();
        writer << "(void** inputs, void** outputs, cublasHandle_t& cublas_handle, "
                  "cudnnHandle_t& "
                  "cudnn_handle)\n";
        writer << "{\n";
        writer.indent++;

        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            const op::Constant* c = dynamic_cast<op::Constant*>(node.get());
            if (c)
            {
                shared_ptr<descriptor::TensorView> tv = node->get_outputs()[0].get_tensor_view();
                writer << "if(" << tv->get_tensor().get_name() << " == NULL)\n";
                writer << "{\n";
                writer.indent++;
                writer << tv->get_tensor().get_name() << " = ("
                       << tv->get_tensor().get_element_type().c_type_string()
                       << " *) runtime::gpu::create_gpu_buffer(" << tv->get_tensor().size()
                       << ");\n";
                writer << "runtime::gpu::cuda_memcpyHtD(" << tv->get_tensor().get_name() << ", "
                       << tv->get_tensor().get_name() << "_cpu, " << tv->get_tensor().size()
                       << ");\n";
                writer.indent--;
                writer << "}\n";
            }
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
            // TODO memory pool malloc.
            writer << "void* pool_base_ptr = ngraph::runtime::gpu::create_gpu_buffer("
                   << temp_pool_size << ");\n";

            // Add temporaries to the variable name map
            for (shared_ptr<Node> node : current_function->get_ordered_ops())
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

        // Add outputs to the variable name map
        output_index = 0;
        for (size_t i = 0; i < current_function->get_output_size(); ++i)
        {
            shared_ptr<Node> op = current_function->get_output_op(i);
            shared_ptr<descriptor::TensorView> tv = op->get_output_tensor_view();
            const element::Type& et = tv->get_tensor_view_type()->get_element_type();
            bool parameter_as_output = false;
            for (shared_ptr<ngraph::op::Parameter> param : current_function->get_parameters())
            {
                for (const descriptor::Output& pout : param->get_outputs())
                {
                    shared_ptr<descriptor::TensorView> ptv = pout.get_tensor_view();
                    if (tv == ptv)
                    {
                        parameter_as_output = true;
                        writer << "ngraph::runtime::gpu::cuda_memcpyDtD(reinterpret_cast<"
                               << et.c_type_string() << "*>(outputs[" << output_index << "]), "
                               << m_variable_name_map[ptv->get_tensor().get_name()] << ", "
                               << ptv->get_tensor().size() << ");\n";
                        break;
                    }
                }
            }
            if (!parameter_as_output && !contains(aliases, output_index))
            {
                if (contains(constants, tv.get()))
                {
                    writer << "ngraph::runtime::gpu::cuda_memcpyHtD(outputs[" << output_index
                           << "], " << tv->get_tensor().get_name() << ", "
                           << tv->get_tensor().size() << ");\n";
                }
                else
                {
                    string type = et.c_type_string();
                    stringstream ss;
                    ss << "((" << type << "*)(outputs[" << output_index << "]))";
                    m_variable_name_map[tv->get_tensor().get_name()] = ss.str();
                }
            }
            output_index++;
        }

        for (shared_ptr<Node> node : current_function->get_ordered_ops())
        {
            auto& n = *node; // Work around a compiler warning (*node inside typeid may have effects
            // with shared pointers, which is fine here but clang doesn't like it.)
            auto handler = dispatcher.find(type_index(typeid(n)));
            if (handler == dispatcher.end())
            {
                throw ngraph_error("Unhandled op during code generation : " + node->description());
            }
            vector<GPU_TensorViewWrapper> in;
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                in.push_back(
                    GPU_TensorViewWrapper(tv, m_variable_name_map[tv->get_tensor().get_name()]));
            }
            vector<GPU_TensorViewWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::TensorView> tv = output.get_tensor_view();
                out.push_back(
                    GPU_TensorViewWrapper(tv, m_variable_name_map[tv->get_tensor().get_name()]));
            }

            // Emit operation prologue
            if (!node->is_parameter() && !node->is_constant())
            {
                if (m_emit_timing)
                {
                    emit_debug_function_entry(writer, node.get(), in, out);
                }
            }

            // Emit operation body
            string func_name;
            auto it = match_functions.find(node.get());
            if (it != match_functions.end())
            {
                func_name = it->second;
            }
            if (func_name.empty())
            {
                handler->second(this, writer, node.get(), in, out);
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
                writer << func_name << "(" << join(names) << ");\n";
            }

            // Emit operation epilogue
            if (!node->is_parameter() && !node->is_constant())
            {
                if (m_emit_timing)
                {
                    emit_debug_function_exit(writer, node.get(), in, out);
                }
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

void runtime::gpu::GPU_ExternalFunction::handle_output_alias(
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
                    writer << "ngraph::runtime::gpu::cuda_memcpyDtD(static_cast<void*>("
                              "outputs["
                           << outputs[i] << "]), static_cast<void*>(outputs[" << outputs[0]
                           << "]), " << otv->get_tensor().size() << ");\n";
                }
                writer.indent--;
                writer << "}\n";
            }
        }
    }
}

shared_ptr<ngraph::runtime::CallFrame> runtime::gpu::GPU_ExternalFunction::make_call_frame()
{
    if (!m_is_compiled)
    {
        compile();
    }

    return make_shared<GPU_CallFrame>(shared_from_this(), m_compiled_function);
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_entry(
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<GPU_TensorViewWrapper>& in,
    const std::vector<GPU_TensorViewWrapper>& out)
{
    writer << "timer_" << node->get_name() << ".start();\n";
}

void runtime::gpu::GPU_ExternalFunction::emit_debug_function_exit(
    codegen::CodeWriter& writer,
    Node* node,
    const std::vector<GPU_TensorViewWrapper>& in,
    const std::vector<GPU_TensorViewWrapper>& out)
{
    writer << "timer_" << node->get_name() << ".stop();\n";
}
