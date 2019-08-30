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

#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Lstm)
            {
                if (!runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error(
                        "Lstm is supported only through MKLDNN and doesnt have reference "
                        "INTERPRETER implementation");
                }
                if (args.size() != 5)
                {
                    throw ngraph_error(
                        "Lstm op doesnt have the required number of inputs to create MKLDNN "
                        "kernel");
                }
                auto& functors = external_function->get_functors();

                auto src_layer_buffer_index =
                    external_function->get_buffer_index(args[0].get_name());
                auto src_iter_buffer_index =
                    external_function->get_buffer_index(args[1].get_name());
                auto weights_layer_buffer_index =
                    external_function->get_buffer_index(args[2].get_name());
                auto weights_iter_buffer_index =
                    external_function->get_buffer_index(args[3].get_name());
                auto bias_buffer_index = external_function->get_buffer_index(args[4].get_name());
                auto dst_layer_buffer_index =
                    external_function->get_buffer_index(out[0].get_name());
                auto dst_iter_buffer_index = external_function->get_buffer_index(out[1].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto lstm_desc =
                    mkldnn_emitter->get_rnn_forward_desc<ngraph::op::Lstm>(node, args, out);
                // Lstm needs 9 primitives: src_layer, src_iter, weights_layer, weights_iter, bias,
                // dst_layer, dst_iter, workspace, and rnn_forward.
                // It needs a new workspace.
                auto lstm_index =
                    mkldnn_emitter->reserve_primitive_space(9, true /* new workspace */);
                auto& deps = mkldnn_emitter->get_primitive_deps(lstm_index);

                auto functor = [&,
                                lstm_desc,
                                lstm_index,
                                src_layer_buffer_index,
                                src_iter_buffer_index,
                                weights_layer_buffer_index,
                                weights_iter_buffer_index,
                                bias_buffer_index,
                                dst_layer_buffer_index,
                                dst_iter_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* ectx) {
                    if (ctx->first_iteration)
                    {
                        mkldnn_emitter->build_rnn_forward(ctx->mkldnn_primitives,
                                                          ctx->mkldnn_workspaces,
                                                          lstm_desc,
                                                          deps,
                                                          lstm_index);
                    }
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[0], ctx->buffer_data[src_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[1], ctx->buffer_data[src_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[2], ctx->buffer_data[weights_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[3], ctx->buffer_data[weights_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[4], ctx->buffer_data[bias_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[5], ctx->buffer_data[dst_layer_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[6], ctx->buffer_data[dst_iter_buffer_index]);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[7], ctx->mkldnn_workspaces[deps[8]]);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, lstm_index);
                };
                functors.emplace_back(functor);
            }

            void register_builders_lstm_cpp() { REGISTER_OP_BUILDER(Lstm); }
        }
    }
}
