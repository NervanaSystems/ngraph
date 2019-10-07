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

#include "ngraph/op/fused/gelu.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Gelu)
            {
                std::cout << "In Builder for Gelu \n";
                auto& functors = external_function->get_functors();

                auto input_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t count = out[0].get_size();

                //auto alpha = static_cast<const ngraph::op::BoundedRelu*>(node)->get_alpha();

                std::cout << "\tuse mkldnn = " << runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node) << "\n";
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    std::cout << "Registering functor for Gelu\n";
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto gelu_desc = mkldnn_emitter->get_gelu_forward_desc(node);
                    QUERY_SCRATCHPAD(eltwise_forward, gelu_desc);

                    // Gelu needs 3 primitives: input, result, and eltwise_forward.
                    auto gelu_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(gelu_index);

                    auto functor = [&,
                                    gelu_desc,
                                    gelu_index,
                                    input_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_gelu(ctx->mkldnn_memories,
                                                               ctx->mkldnn_primitives,
                                                               ctx->mkldnn_scratchpad_mds,
                                                               gelu_desc,
                                                               deps,
                                                               gelu_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[input_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx, gelu_index, deps, cpu::mkldnn_utils::OpType::GELU);
                    };
                    std::cout << "Registered functor for Gelu\n";
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error(
                        "Gelu is supported with MKLDNN kernel only for f32.");
                    /*std::function<decltype(runtime::cpu::kernel::bounded_relu<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::bounded_relu)

                    auto functor = [&, kernel, alpha, count, input_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[input_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               alpha,
                               count,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);*/
                }
            }


            /*template <>
            void Builder::BUILDER_DECL(ngraph::op::GeluBackpropFactor)
            {
                std::cout << "GeluBackpropFactor builder begin\n";
                auto& functors = external_function->get_functors();

                auto arg_fwd_buffer_index = external_function->get_buffer_index(args[0].get_name());
                //auto delta_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t count = out[0].get_size();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    std::cout << "GeluBackpropFactor builder mkldnn true\n";
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_gelu_backward_desc(node);
                    std::cout << "GeluBackpropFactor builder after backward desc\n";
                    auto fwd_desc = mkldnn_emitter->get_gelu_forward_desc(node);
                    std::cout << "GeluBackpropFactor builder after forward desc\n";
                    QUERY_SCRATCHPAD_2ARGS(eltwise_backward, fwd_desc, bwd_desc);

                    // geluBackprop needs 3 primitives: input, result, and eltwise_backward.
                    size_t gelu_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(gelu_index);

                    auto functor = [&,
                                    bwd_desc,
                                    fwd_desc,
                                    gelu_index,
                                    arg_fwd_buffer_index,
                                    //delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx ) {
                        std::cout << "GeluBackpropFactor builder inside functor\n";
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_gelu_backward(ctx->mkldnn_memories,
                                                                ctx->mkldnn_primitives,
                                                                ctx->mkldnn_scratchpad_mds,
                                                                bwd_desc,
                                                                fwd_desc,
                                                                deps,
                                                                gelu_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_fwd_buffer_index]);
                        //cpu::mkldnn_utils::set_memory_ptr(
                        //    ctx, deps[1], ctx->buffer_data[delta_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx, gelu_index, deps, cpu::mkldnn_utils::OpType::GELUBACKPROP);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::cout << "GeluBackpropFactor builder NOT mkldnn\n";
                    throw ngraph_error(
                        "GeluBackpropFactor is supported with MKLDNN kernel only for f32.");
                }
            }*/

            template <>
            void Builder::BUILDER_DECL(ngraph::op::GeluBackprop)
            {
                std::cout << "GeluBackprop builder begin\n";
                auto& functors = external_function->get_functors();

                auto arg_fwd_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto delta_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t count = out[0].get_size();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    std::cout << "GeluBackprop builder mkldnn true\n";
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto bwd_desc = mkldnn_emitter->get_gelu_backward_desc(node);
                    auto fwd_desc = mkldnn_emitter->get_gelu_forward_desc(node);
                    QUERY_SCRATCHPAD_2ARGS(eltwise_backward, fwd_desc, bwd_desc);

                    // geluBackprop needs 4 primitives: input, delta, result, and eltwise_backward.
                    size_t gelu_b_index = mkldnn_emitter->reserve_primitive_space(4);
                    auto& deps = mkldnn_emitter->get_primitive_deps(gelu_b_index);

                    auto functor = [&,
                                    bwd_desc,
                                    fwd_desc,
                                    gelu_b_index,
                                    arg_fwd_buffer_index,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        std::cout << "GeluBackpropFactor builder inside functor\n";
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_gelu_backward(ctx->mkldnn_memories,
                                                                ctx->mkldnn_primitives,
                                                                ctx->mkldnn_scratchpad_mds,
                                                                bwd_desc,
                                                                fwd_desc,
                                                                deps,
                                                                gelu_b_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_fwd_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[delta_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx, gelu_b_index, deps, cpu::mkldnn_utils::OpType::GELUBACKPROP);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::cout << "GeluBackprop builder NOT mkldnn\n";
                    throw ngraph_error(
                        "GeluBackprop is supported with MKLDNN kernel only for f32.");
                    // call the reference implementation???
                    /*std::function<decltype(runtime::cpu::kernel::gelu_backprop<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::gelu_backprop)

                    auto functor = [&,
                                    kernel,
                                    count,
                                    arg_fwd_buffer_index,
                                    delta_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_fwd_buffer_index],
                               ctx->buffer_data[delta_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               count,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);*/
                }
            }

            void register_builders_gelu_cpp()
            {
                REGISTER_OP_BUILDER(Gelu);
                REGISTER_OP_BUILDER(GeluBackprop);
            }

        }
    }
}
