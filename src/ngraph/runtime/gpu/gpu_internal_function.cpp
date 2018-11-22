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
#include "ngraph/runtime/gpu/gpu_internal_function.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"
#include "ngraph/runtime/gpu/op/batch_norm.hpp"
#include "ngraph/runtime/gpu/op/rnn.hpp"
#include "ngraph/runtime/gpu/pass/gpu_batch_norm_cache.hpp"
#include "ngraph/runtime/gpu/pass/gpu_layout.hpp"
#include "ngraph/runtime/gpu/pass/gpu_rnn_fusion.hpp"
#include "ngraph/runtime/gpu/pass/tensor_memory_reservation.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_constructor.hpp"
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_invoke.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

using namespace std;
using namespace ngraph;

static std::mutex s_compilation;

std::string runtime::gpu::GPU_InternalFunction::emit_op(GPU_CompiledFunction* compiled_function,
                                                        const ngraph::Node* node,
                                                        const std::vector<GPUTensorWrapper>& args,
                                                        const std::vector<GPUTensorWrapper>& out)
{
    auto emit_function = GPU_Emitter::get_emit_function(*node);
    return emit_function(compiled_function, node, args, out);
};

runtime::gpu::GPU_InternalFunction::GPU_InternalFunction(
    const shared_ptr<ngraph::Function>& function,
    std::shared_ptr<GPU_Backend::BackendContext>& shared_context)
    : GPU_CompiledFunction(function, shared_context)
{
}

runtime::gpu::GPU_InternalFunction::~GPU_InternalFunction()
{
    if (m_trace)
    {
        string filename = file_util::path_join(s_output_dir, m_function_name + "_trace.txt");
        ofstream out(filename);
        out << m_trace->get_code();
        out.close();
    }
}

std::string runtime::gpu::GPU_InternalFunction::add_to_runtime(size_t primitive_index,
                                                               const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                                               const std::vector<runtime::gpu::GPUTensorWrapper>& out)
{
    std::function<void(GPUCallFrame& call_frame, GPURuntimeContext* ctx)> primitive_invocation;
    if (m_trace)
    {
        primitive_invocation = [this, args, out, primitive_index](GPUCallFrame& call_frame, GPURuntimeContext* ctx) mutable
        {
            // here, these inputs and outputs could be any of [constant, input, output, intermediate]
            auto inputs = call_frame.get_tensor_io(args);
            auto outputs = call_frame.get_tensor_io(out);
            *m_trace << "(";
            for (size_t i = 0; i < outputs.size(); i++)
            {
                if (i != 0)
                {
                    *m_trace << ", ";
                }
                *m_trace << std::hex << outputs[i];
            }
            *m_trace << ") = primitive(" << primitive_index << ", ";
            for (size_t i = 0; i < inputs.size(); i++)
            {
                if (i != 0)
                {
                    *m_trace << ", ";
                }
                *m_trace << std::hex << inputs[i];
            }
            *m_trace << ");\n";
            *m_trace << compose_manifest(primitive_index, args, out);
            runtime::gpu::invoke_primitive(ctx, primitive_index, inputs.data(), outputs.data());
        };
    }
    else
    {
        primitive_invocation = [args, out, primitive_index](GPUCallFrame& call_frame, GPURuntimeContext* ctx) mutable
        {
            // here, these inputs and outputs could be any of [constant, input, output, intermediate]
            auto inputs = call_frame.get_tensor_io(args);
            auto outputs = call_frame.get_tensor_io(out);
            runtime::gpu::invoke_primitive(ctx, primitive_index, inputs.data(), outputs.data());
        };
    }
    m_runtime_constructor->add(primitive_invocation);

    return compose_manifest(primitive_index, args, out);
}

std::string runtime::gpu::GPU_InternalFunction::compose_manifest(const size_t& primitive_index,
                                                                 const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                                                 const std::vector<runtime::gpu::GPUTensorWrapper>& out) const
{
    codegen::CodeWriter writer;
    writer.block_begin();
    {
        for (auto const& tensor : args)
        {
            writer << "push " << tensor << "\n";
        }
        writer << "call primitive(" << primitive_index << ");\n";
        for (auto const& tensor : out)
        {
            writer << "pop " << tensor << "\n";
        }
    }
    writer.block_end();
    return writer.get_code();
}


void runtime::gpu::GPU_InternalFunction::build_functions()
{
    for (const auto& p : m_function_ordered_ops)
    {
        auto& current_function = p.first;
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
                m_variable_name_map[tv->get_name()] =
                    std::make_tuple(runtime::gpu::GPUTensorWrapper::TensorType::INPUT, arg_index, ss.str());
                // propagate_in_place_input(&param->get_outputs().at(i), ss.str());
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
            m_variable_name_map[tv->get_name()] =
                std::make_tuple(runtime::gpu::GPUTensorWrapper::TensorType::OUTPUT, i, ss.str());

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
                m_variable_name_map[itv->get_name()] =
                    std::make_tuple(runtime::gpu::GPUTensorWrapper::TensorType::OUTPUT, i, ss.str());
                //propagate_in_place_output(&(res->get_inputs().at(0).get_output()), output_name);
            }
        }

        // Add temporaries to the variable name map
        bool temporaries_used = false;
        for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
        {
            if (node->liveness_new_list.size() > 0)
            {
                temporaries_used = true;
                break;
            }
        }
        if (temporaries_used)
        {
            for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
            {
                for (descriptor::Tensor* tensor : node->liveness_new_list)
                {
                    m_variable_name_map[tensor->get_name()] =
                        std::make_tuple(runtime::gpu::GPUTensorWrapper::TensorType::INTERMEDIATE,
                                        tensor->get_pool_offset(),
                                        current_function->get_name());
                }
            }
        }

        // Add constants to the variable name map
        for (shared_ptr<Node> node : p.second)
        {
            if (auto c = std::dynamic_pointer_cast<op::Constant>(node))
            {
                shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
                m_variable_name_map[tv->get_name()] =
                    std::make_tuple(runtime::gpu::GPUTensorWrapper::TensorType::CONSTANT, 0,
                                    node->get_name());
            }
        }

        for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
        {
            vector<string> node_input_names;
            vector<string> node_output_names;
            vector<GPUTensorWrapper> in;
            for (const descriptor::Input& input : node->get_inputs())
            {
                const descriptor::Output& output = input.get_output();
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                auto& var = m_variable_name_map[tv->get_name()];
                in.push_back(GPUTensorWrapper(tv, std::get<0>(var), std::get<1>(var), std::get<2>(var)));
                node_input_names.emplace_back(tv->get_name());
            }
            vector<GPUTensorWrapper> out;
            for (const descriptor::Output& output : node->get_outputs())
            {
                shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
                auto& var = m_variable_name_map[tv->get_name()];
                out.push_back(GPUTensorWrapper(tv, std::get<0>(var), std::get<1>(var), std::get<2>(var)));
                node_output_names.emplace_back(tv->get_name());
            }

            // Emit function description comment
            if (!node->is_parameter() && !node->is_constant())
            {
                m_manifest << "\n// " << current_function->get_name() << "::" << node->get_name() << "(";
                vector<string> parameter_nodes = node_input_names;
                parameter_nodes.insert(
                    parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
                m_manifest << join(parameter_nodes);
                m_manifest << ")\n";
                // emit_debug_function_entry(node.get());
            }

            // Emit operation body
            // m_writer << emit_op(this, node.get(), in, out);
            m_manifest << emit_op(this, node.get(), in, out);

            // Emit operation epilogue
            // if (!node->is_parameter() && !node->is_constant())
            // {
            //     emit_debug_function_exit(node.get());
            // }
        }

    }
}

void runtime::gpu::GPU_InternalFunction::compile()
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
    string dump_filename = file_util::path_join(s_output_dir, m_function_name + "_ops.txt");
    pass_manager.register_pass<ngraph::pass::DumpSorted>(dump_filename);
    pass_manager.run_passes(m_function);

    for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
    {
        m_function_ordered_ops.emplace(current_function, current_function->get_ordered_ops());
    }

    m_runtime_constructor = runtime::gpu::make_unique<GPURuntimeConstructor>(m_function_ordered_ops);

    if (std::getenv("NGRAPH_GPU_TRACE"))
    {
        m_trace = std::make_shared<codegen::CodeWriter>();
    }

    // build and emit functions
    build_functions();

    // allocate device buffers for intermediates, primitive arguments, and workspace
    allocator->close();
    m_shared_context->m_primitive_emitter->allocate_primitive_memory();

    GPUCallFrame call_frame(m_function->get_parameters().size(), m_function->get_output_size());

    // resolve memory reservations (constants and intermediate buffers)
    call_frame.resolve_reservations(this, m_tensor_memory_buffers);

    // build runtime
    m_runtime = m_runtime_constructor->build(call_frame);

    // store manifest
    save_manifest_to_disk();

    m_is_compiled = true;
}

void runtime::gpu::GPU_InternalFunction::save_manifest_to_disk() const
{
    string filename = file_util::path_join(s_output_dir, m_function_name + "_manifest.txt");
    ofstream out(filename);
    out << m_manifest.get_code();
    out.close();
}

void runtime::gpu::GPU_InternalFunction::propagate_in_place_input(
    ngraph::descriptor::Output* output, std::string input_name)
{
    // std::deque<ngraph::descriptor::Output*> stack;
    // stack.push_front(output);

    // while (stack.size() > 0)
    // {
    //     ngraph::descriptor::Output* it = stack.front();
    //     stack.pop_front();
    //     for (auto input : it->get_inputs())
    //     {
    //         auto c_op = std::dynamic_pointer_cast<ngraph::op::Op>(input->get_node());
    //         if (!c_op || c_op->is_output())
    //         {
    //             continue;
    //         }

    //         if (auto op_annotations = c_op->get_op_annotations())
    //         {
    //             for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
    //             {
    //                 if (oi_pair.input == input->get_index() && !oi_pair.destructive)
    //                 {
    //                     size_t output_index = oi_pair.output;
    //                     auto& output_tensor = c_op->get_outputs().at(output_index).get_tensor();

    //                     m_variable_name_map[output_tensor.get_name()] = input_name;

    //                     NGRAPH_DEBUG << "GPU codegen: Forwarding " << input_name << " through "
    //                                  << output_tensor.get_name();
    //                     stack.push_back(&c_op->get_outputs().at(output_index));
    //                 }
    //             }
    //         }
    //     }
    // }
}

void runtime::gpu::GPU_InternalFunction::propagate_in_place_output(
    ngraph::descriptor::Output* res_src_output, std::string output_name)
{
    // // we start with a particular output
    // // which is an argument to a given op::Result
    // size_t offset = res_src_output->get_tensor().get_pool_offset();
    // auto it = res_src_output;

    // bool propagate_further = false;
    // do
    // {
    //     propagate_further = false;
    //     auto arg = std::dynamic_pointer_cast<ngraph::op::Op>(it->get_node());
    //     if (!arg)
    //     {
    //         break;
    //     }
    //     if (auto op_annotations = arg->get_op_annotations())
    //     {
    //         for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
    //         {
    //             if (oi_pair.output == it->get_index())
    //             {
    //                 size_t input_index = oi_pair.input;
    //                 auto& input_tensor = arg->get_inputs().at(input_index).get_tensor();
    //                 auto tmp_node = arg->get_inputs().at(input_index).get_output().get_node();
    //                 if (input_tensor.get_pool_offset() == offset && !tmp_node->is_parameter() &&
    //                     !tmp_node->is_constant())
    //                 {
    //                     NGRAPH_DEBUG << "Reusing " << output_name << " for "
    //                                  << input_tensor.get_name();

    //                     m_variable_name_map[input_tensor.get_name()] = output_name;

    //                     it = &arg->get_inputs().at(input_index).get_output();
    //                     propagate_further = true;
    //                 }
    //             }
    //         }
    //     }
    // } while (propagate_further);
}

void runtime::gpu::GPU_InternalFunction::get_performance_data(std::vector<runtime::PerformanceCounter>& rc) const
{
    // auto* engine = this->m_execution_engine.get();
    // if (engine)
    // {
    //     auto get_count = engine->find_function<size_t()>("get_debug_timer_count");
    //     auto get_name = engine->find_function<const char*(size_t)>("get_debug_timer_name");
    //     auto get_microseconds =
    //         engine->find_function<size_t(size_t)>("get_debug_timer_microseconds");
    //     auto get_call_count =
    //         engine->find_function<size_t(size_t)>("get_debug_timer_call_count");

    //     if (get_count && get_name && get_microseconds && get_call_count)
    //     {
    //         size_t count = get_count();
    //         for (size_t i = 0; i < count; i++)
    //         {
    //             rc.push_back({get_name(i), get_microseconds(i), get_call_count(i)});
    //         }
    //     }
    // }
}
