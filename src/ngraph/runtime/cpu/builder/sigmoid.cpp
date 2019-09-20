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

#include "ngraph/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/sigmoid_multiply.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Sigmoid)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto input_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto sigmoid_desc = mkldnn_emitter->get_sigmoid_forward_desc(node, false);
                size_t scratchpad_size = QUERY_SCRATCHPAD(eltwise_forward, sigmoid_desc);

                // Sigmoid needs 3 primitives: input, result, and eltwise_forward.
                auto sigmoid_index = mkldnn_emitter->reserve_primitive_space(3);
                auto& deps = mkldnn_emitter->get_primitive_deps(sigmoid_index);

                auto functor = [&,
                                sigmoid_desc,
                                sigmoid_index,
                                scratchpad_size,
                                arg0_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* /* ectx */) {
                    if (ctx->first_iteration)
                    {
                        mkldnn_emitter->build_sigmoid_forward(ctx->mkldnn_memories,
                                                              ctx->mkldnn_primitives,
                                                              ctx->mkldnn_scratchpad_mds,
                                                              sigmoid_desc,
                                                              deps,
                                                              sigmoid_index);
                    }
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx,
                                                               sigmoid_index,
                                                               deps,
                                                               cpu::mkldnn_utils::OpType::SIGMOID,
                                                               scratchpad_size);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::SigmoidBackprop)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto input_shape = args[0].get_shape();
                auto delta_shape = args[1].get_shape();
                auto out_shape = out[0].get_shape();

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto fwd_desc = mkldnn_emitter->get_sigmoid_forward_desc(node, true);
                auto bwd_desc = mkldnn_emitter->get_sigmoid_backward_desc(node);
                size_t scratchpad_size =
                    QUERY_SCRATCHPAD_2ARGS(eltwise_backward, fwd_desc, bwd_desc);

                // SigmoidBackprop needs 4 primitives: input, delta, result, and eltwise_backward.
                size_t sigmoid_index = mkldnn_emitter->reserve_primitive_space(4);
                auto& deps = mkldnn_emitter->get_primitive_deps(sigmoid_index);

                auto functor = [&,
                                bwd_desc,
                                fwd_desc,
                                sigmoid_index,
                                scratchpad_size,
                                arg0_buffer_index,
                                arg1_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* /* ectx */) {
                    if (ctx->first_iteration)
                    {
                        mkldnn_emitter->build_sigmoid_backward(ctx->mkldnn_memories,
                                                               ctx->mkldnn_primitives,
                                                               ctx->mkldnn_scratchpad_mds,
                                                               bwd_desc,
                                                               fwd_desc,
                                                               deps,
                                                               sigmoid_index);
                    }
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[arg0_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[arg1_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[out_buffer_index]);

                    cpu::mkldnn_utils::mkldnn_invoke_primitive(
                        ctx,
                        sigmoid_index,
                        deps,
                        cpu::mkldnn_utils::OpType::SIGMOIDBACKPROP,
                        scratchpad_size);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::SigmoidMultiply)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto buffer_index_size = shape_size(args[0].get_shape());

                auto sigmoid_mul = static_cast<const ngraph::op::SigmoidMultiply*>(node);

                const size_t index =
                    static_cast<size_t>(sigmoid_mul->get_input_func_type(0)) *
                        static_cast<size_t>(ngraph::op::SigmoidMultiply::FunctionType::NumTypes) +
                    static_cast<size_t>(sigmoid_mul->get_input_func_type(1));

                auto functor = [&,
                                index,
                                buffer_index_size,
                                arg0_buffer_index,
                                arg1_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                    ngraph::runtime::cpu::kernel::sigmoid_multiply(
                        ctx->buffer_data[arg0_buffer_index],
                        ctx->buffer_data[arg1_buffer_index],
                        ctx->buffer_data[out_buffer_index],
                        buffer_index_size,
                        index,
                        ectx->arena);
                };

                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::SigmoidMultiplyBackprop)
            {
                auto& functors = external_function->get_functors();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());
                auto buffer_index_size = shape_size(args[0].get_shape());

                auto sigmoid_mul = static_cast<const ngraph::op::SigmoidMultiplyBackprop*>(node);

                const size_t index =
                    static_cast<size_t>(sigmoid_mul->get_input_func_type(0)) *
                        static_cast<size_t>(ngraph::op::SigmoidMultiply::FunctionType::NumTypes) +
                    static_cast<size_t>(sigmoid_mul->get_input_func_type(1));

                auto functor = [&,
                                index,
                                buffer_index_size,
                                arg0_buffer_index,
                                arg1_buffer_index,
                                arg2_buffer_index,
                                out0_buffer_index,
                                out1_buffer_index](CPURuntimeContext* ctx,
                                                   CPUExecutionContext* ectx) {
                    ngraph::runtime::cpu::kernel::sigmoid_multiply_backprop(
                        ctx->buffer_data[arg0_buffer_index],
                        ctx->buffer_data[arg1_buffer_index],
                        ctx->buffer_data[arg2_buffer_index],
                        ctx->buffer_data[out0_buffer_index],
                        ctx->buffer_data[out1_buffer_index],
                        buffer_index_size,
                        index,
                        ectx->arena);
                };

                functors.emplace_back(functor);
            }

            void register_builders_sigmoid_cpp()
            {
                REGISTER_OP_BUILDER(Sigmoid);
                REGISTER_OP_BUILDER(SigmoidBackprop);
                REGISTER_OP_BUILDER(SigmoidMultiply);
                REGISTER_OP_BUILDER(SigmoidMultiplyBackprop);
            }
        }
    }
}
