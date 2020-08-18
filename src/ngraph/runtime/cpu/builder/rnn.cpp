//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Rnn)
            {
                if (!runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    throw ngraph_error(
                        "Rnn is supported only through DNNL and doesnt have reference "
                        "INTERPRETER implementation");
                }

                auto& functors = external_function->get_functors();
                auto rnn_op = static_cast<const ngraph::op::Rnn*>(node);
                auto src_layer_buffer_index =
                    external_function->get_buffer_index(args[0].get_name());
                auto src_iter_buffer_index =
                    external_function->get_buffer_index(args[1].get_name());
                auto dst_layer_buffer_index =
                    external_function->get_buffer_index(out[0].get_name());
                auto dst_iter_buffer_index = external_function->get_buffer_index(out[1].get_name());

                auto& dnnl_emitter = external_function->get_dnnl_emitter();

                if (rnn_op->is_type(ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_rnn))
                {
                    auto weights_layer_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto weights_iter_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto bias_buffer_index =
                        external_function->get_buffer_index(args[4].get_name());

                    // Rnn needs 9 primitives: src_layer, src_iter, weights_layer, weights_iter,
                    // bias,
                    // dst_layer, dst_iter, workspace, and rnn_forward.
                    // It needs a new workspace.
                    auto rnn_index = dnnl_emitter->reserve_primitive_space(
                        9, false /* fwd and bwd */, true /* new workspace */);
                    auto& deps = dnnl_emitter->get_primitive_deps(rnn_index);
                    auto vanilla_rnn_desc =
                        dnnl_emitter->get_vanilla_rnn_forward_desc<ngraph::op::Rnn>(
                            node, args, out);
                    size_t scratchpad_size =
                        dnnl_emitter->query_scratchpad_vanilla_rnn_forward(vanilla_rnn_desc);

                    auto functor = [&,
                                    vanilla_rnn_desc,
                                    rnn_index,
                                    scratchpad_size,
                                    src_layer_buffer_index,
                                    src_iter_buffer_index,
                                    weights_layer_buffer_index,
                                    weights_iter_buffer_index,
                                    bias_buffer_index,
                                    dst_layer_buffer_index,
                                    dst_iter_buffer_index](CPURuntimeContext* ctx,
                                                           CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_vanilla_rnn_forward(ctx->dnnl_memories,
                                                                    ctx->dnnl_primitives,
                                                                    ctx->dnnl_scratchpad_mds,
                                                                    ctx->dnnl_workspaces,
                                                                    vanilla_rnn_desc,
                                                                    deps,
                                                                    rnn_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[src_layer_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[src_iter_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[weights_layer_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[weights_iter_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[4], ctx->buffer_data[bias_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[5], ctx->buffer_data[dst_layer_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[6], ctx->buffer_data[dst_iter_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[7], ctx->dnnl_workspaces[deps[8]]);
                        cpu::dnnl_utils::dnnl_invoke_primitive(ctx,
                                                               rnn_index,
                                                               deps,
                                                               cpu::dnnl_utils::OpType::VANILLA_RNN,
                                                               scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else if (rnn_op->is_type(ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm))
                {
                    auto src_iter_c_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto weights_layer_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto weights_iter_buffer_index =
                        external_function->get_buffer_index(args[4].get_name());
                    auto bias_buffer_index =
                        external_function->get_buffer_index(args[5].get_name());
                    auto dst_iter_c_buffer_index =
                        external_function->get_buffer_index(out[2].get_name());

                    // Rnn needs 11 primitives: src_layer, src_iter, src_iter_c, weights_layer,
                    // weights_iter, bias,
                    // dst_layer, dst_iter, dst_iter_c, workspace, and lstm_forward.
                    // It needs a new workspace.
                    auto rnn_index = dnnl_emitter->reserve_primitive_space(
                        11, false /* fwd and bwd */, true /* new workspace */);
                    auto& deps = dnnl_emitter->get_primitive_deps(rnn_index);
                    auto rnn_desc =
                        dnnl_emitter->get_rnn_forward_desc<ngraph::op::Rnn>(node, args, out);
                    size_t scratchpad_size = dnnl_emitter->query_scratchpad_rnn_forward(rnn_desc);

                    auto functor = [&,
                                    rnn_desc,
                                    rnn_index,
                                    scratchpad_size,
                                    src_layer_buffer_index,
                                    src_iter_buffer_index,
                                    src_iter_c_buffer_index,
                                    weights_layer_buffer_index,
                                    weights_iter_buffer_index,
                                    bias_buffer_index,
                                    dst_layer_buffer_index,
                                    dst_iter_buffer_index,
                                    dst_iter_c_buffer_index](CPURuntimeContext* ctx,
                                                             CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_rnn_forward(ctx->dnnl_memories,
                                                            ctx->dnnl_primitives,
                                                            ctx->dnnl_scratchpad_mds,
                                                            ctx->dnnl_workspaces,
                                                            rnn_desc,
                                                            deps,
                                                            rnn_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[src_layer_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[src_iter_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[src_iter_c_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[weights_layer_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[4], ctx->buffer_data[weights_iter_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[5], ctx->buffer_data[bias_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[6], ctx->buffer_data[dst_layer_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[7], ctx->buffer_data[dst_iter_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[8], ctx->buffer_data[dst_iter_c_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[9], ctx->dnnl_workspaces[deps[10]]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(
                            ctx, rnn_index, deps, cpu::dnnl_utils::OpType::RNN, scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
            }

            void register_builders_rnn_cpp() { REGISTER_OP_BUILDER(ngraph::op::Rnn); }
        }
    }
}
