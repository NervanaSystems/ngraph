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

using namespace std;
using namespace ngraph;

static std::mutex s_compilation;

// std::string runtime::gpu::GPU_InternalFunction::emit_op(GPU_InternalFunction* external_function,
//                                                         const ngraph::Node* node,
//                                                         const std::vector<GPUTensorWrapper>& args,
//                                                         const std::vector<GPUTensorWrapper>& out)
// {
//     auto emit_function = GPU_Emitter::get_emit_function(*node);
//     return emit_function(external_function, node, args, out);
// };

runtime::gpu::GPU_InternalFunction::GPU_InternalFunction(
    const shared_ptr<ngraph::Function>& function,
    std::shared_ptr<GPU_Backend::BackendContext>& shared_context)
    : GPU_CompiledFunction(function, shared_context)
{
}

runtime::gpu::GPU_InternalFunction::~GPU_InternalFunction()
{
}

std::string runtime::gpu::GPU_InternalFunction::add_to_runtime(size_t primitive_index,
                                                               const std::vector<runtime::gpu::GPUTensorWrapper>& args,
                                                               const std::vector<runtime::gpu::GPUTensorWrapper>& out)
{
    // codegen::CodeWriter writer;
    // writer << "void* input[] = {" << node_names(args) << "};\n";
    // writer << "void* output[] = {" << node_names(out) << "};\n";
    // writer << "gpu::invoke_primitive(ctx, " << primitive_index << ", input, output);\n";
    // return writer.get_code();
    return "";
}

// void runtime::gpu::GPU_InternalFunction::emit_constant_declarations()
// {
//     m_writer << "// Declare all constants\n";
//     for (const auto& p : m_function_ordered_ops)
//     {
//         for (shared_ptr<Node> node : p.second)
//         {
//             const op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
//             if (c)
//             {
//                 shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
//                 // get an allocator for transient per kernel gpu memory
//                 runtime::gpu::GPUAllocator allocator =
//                     m_shared_context->m_primitive_emitter->get_memory_allocator();
//                 size_t idx = allocator.reserve_argspace(c->get_data_ptr(),
//                                                         tv->size() * tv->get_element_type().size());
//                 m_writer << "static size_t " << tv->get_name() << "_idx = " << idx << ";\n";
//                 m_writer << "static " << tv->get_element_type().c_type_string() << "* "
//                          << tv->get_name() << " = nullptr;\n";
//                 m_variable_name_map[tv->get_name()] = tv->get_name();
//             }
//         }
//     }

//     m_writer << "\nstatic bool is_constant_mem_ptr_null = true;\n\n";
//     m_writer << "static void invoke_constant_mem_ptr()\n";
//     m_writer.block_begin();
//     {
//         m_writer << "if(is_constant_mem_ptr_null)\n";
//         m_writer.block_begin();
//         {
//             for (const auto& p : m_function_ordered_ops)
//             {
//                 for (shared_ptr<Node> node : p.second)
//                 {
//                     const op::Constant* c = dynamic_cast<ngraph::op::Constant*>(node.get());
//                     if (c)
//                     {
//                         shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
//                         m_writer << tv->get_name() << " = reinterpret_cast<"
//                                  << tv->get_element_type().c_type_string()
//                                  << "*>(runtime::gpu::invoke_memory_primitive(m_runtime_context, "
//                                  << tv->get_name() << "_idx));\n";
//                     }
//                 }
//             }
//             m_writer << "is_constant_mem_ptr_null = false;\n";
//         }
//         m_writer.block_end();
//     }
//     m_writer.block_end();
// }

// void runtime::gpu::GPU_InternalFunction::emit_functions()
// {
//     for (const auto& p : m_function_ordered_ops)
//     {
//         auto current_function = p.first;
//         set<string> output_names;
//         for (shared_ptr<Node> op : current_function->get_results())
//         {
//             shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
//             output_names.insert(tv->get_name());
//         }
//         set<descriptor::Tensor*> constants;
//         for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
//         {
//             if (dynamic_cast<ngraph::op::Constant*>(node.get()))
//             {
//                 shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
//                 constants.insert(tv.get());
//             }
//         }

//         m_writer << "extern \"C\" void " << current_function->get_name();
//         m_writer << "(void** inputs, void** outputs, "
//                  << "gpu::GPURuntimeContext* ctx) __attribute__ ((optnone))\n";
//         m_writer.block_begin();
//         {
//             m_writer << "m_runtime_context = ctx;\n";
//             // set constant pointers during the first run
//             m_writer << "invoke_constant_mem_ptr();\n";

//             // alocate temp memory pool
//             emit_temp_mem_pool_allocation(current_function);

//             // Add inputs to the variable name map
//             size_t arg_index = 0;
//             for (shared_ptr<ngraph::op::Parameter> param : current_function->get_parameters())
//             {
//                 for (size_t i = 0; i < param->get_output_size(); ++i)
//                 {
//                     shared_ptr<descriptor::Tensor> tv = param->get_output_tensor_ptr(i);
//                     const element::Type& et = tv->get_element_type();
//                     string type = et.c_type_string();
//                     stringstream ss;
//                     ss << "((" << type << "*)(inputs[" << arg_index << "]))";
//                     m_variable_name_map[tv->get_name()] = ss.str();
//                     propagate_in_place_input(&param->get_outputs().at(i), ss.str());
//                     arg_index++;
//                 }
//             }

//             // Add outputs to the variable name map
//             for (size_t i = 0; i < current_function->get_output_size(); ++i)
//             {
//                 shared_ptr<Node> op = current_function->get_output_op(i);
//                 shared_ptr<descriptor::Tensor> tv = op->get_output_tensor_ptr();
//                 string type = tv->get_element_type().c_type_string();
//                 stringstream ss;
//                 ss << "((" << type << "*)(outputs[" << i << "]))";
//                 m_variable_name_map[tv->get_name()] = ss.str();

//                 auto res = dynamic_pointer_cast<ngraph::op::Result>(op);
//                 //keep assigning different outputs to a result descriptor
//                 //op::Result emitter will check if in and out descriptors are the same
//                 //and skip a copy
//                 auto input_node = res->get_inputs().at(0).get_output().get_node();
//                 if (!input_node->is_constant() && !input_node->is_parameter())
//                 {
//                     shared_ptr<descriptor::Tensor> itv =
//                         res->get_inputs().at(0).get_output().get_tensor_ptr();
//                     auto output_name = ss.str();
//                     m_variable_name_map[itv->get_name()] = output_name;
//                     propagate_in_place_output(&(res->get_inputs().at(0).get_output()), output_name);
//                 }
//             }

//             for (shared_ptr<Node> node : m_function_ordered_ops.at(current_function))
//             {
//                 vector<GPUTensorWrapper> in;
//                 vector<string> node_input_names;
//                 vector<string> node_output_names;
//                 for (const descriptor::Input& input : node->get_inputs())
//                 {
//                     const descriptor::Output& output = input.get_output();
//                     shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
//                     in.push_back(GPUTensorWrapper(tv, m_variable_name_map[tv->get_name()]));
//                     node_input_names.emplace_back(tv->get_name());
//                 }
//                 vector<GPUTensorWrapper> out;
//                 for (const descriptor::Output& output : node->get_outputs())
//                 {
//                     shared_ptr<descriptor::Tensor> tv = output.get_tensor_ptr();
//                     out.push_back(GPUTensorWrapper(tv, m_variable_name_map[tv->get_name()]));
//                     node_output_names.emplace_back(tv->get_name());
//                 }

//                 // Emit function description comment
//                 if (!node->is_parameter() && !node->is_constant())
//                 {
//                     m_writer << "\n// " << node->get_name() << "(";
//                     vector<string> parameter_nodes = node_input_names;
//                     parameter_nodes.insert(
//                         parameter_nodes.end(), node_output_names.begin(), node_output_names.end());
//                     m_writer << join(parameter_nodes);
//                     m_writer << ")\n";
//                     emit_debug_function_entry(node.get());
//                 }

//                 // Emit operation body
//                 auto it = m_node_function_map.find(node.get());
//                 if (it == m_node_function_map.end())
//                 {
//                     m_writer << emit_op(this, node.get(), in, out);
//                 }
//                 else
//                 {
//                     string func_name =
//                         ngraph::pass::CommonFunctionCollection::create_function_name(*it->second);
//                     vector<string> names;
//                     for (const GPUTensorWrapper& tv : in)
//                     {
//                         names.push_back(tv.get_name());
//                     }
//                     for (const GPUTensorWrapper& tv : out)
//                     {
//                         names.push_back(tv.get_name());
//                     }
//                     names.push_back("ctx");
//                     m_writer << func_name << "(" << join(names) << ");\n";
//                 }

//                 // Emit operation epilogue
//                 if (!node->is_parameter() && !node->is_constant())
//                 {
//                     emit_debug_function_exit(node.get());
//                 }
//             }
//         }
//         m_writer.block_end(); // End generated function
//     }
// }

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

    // build and emit functions
    // emit_functions();

    // allocate device buffers for primitive arguments and workspace
    allocator->close();
    m_shared_context->m_primitive_emitter->allocate_primitive_memory();

    // store manifest
    // string code = writer.get_code();
    // store_emitted_functions(code);

    // assembly entry point

    // compilation is finished
    m_is_compiled = true;
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
