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

#include "ngraph/op/lrn.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/reference/lrn.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::LRN)
            {
                auto& functors = external_function->get_functors();

                const ngraph::op::LRN* lrn = static_cast<const ngraph::op::LRN*>(node);
                CPUKernelFunctor functor;

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                AxisSet axes = lrn->get_reduction_axes();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto lrn_desc = mkldnn_emitter->get_lrn_forward_desc(node);
                    size_t scratchpad_size = QUERY_SCRATCHPAD(lrn_forward, lrn_desc);

                    // LRN needs 3 primitives: input, result, and lrn_forward.
                    auto lrn_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(lrn_index);

                    functor = [&,
                               lrn_desc,
                               lrn_index,
                               scratchpad_size,
                               arg_buffer_index,
                               out_buffer_index](CPURuntimeContext* ctx,
                                                 CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_lrn_forward(ctx->mkldnn_memories,
                                                              ctx->mkldnn_primitives,
                                                              ctx->mkldnn_scratchpad_mds,
                                                              lrn_desc,
                                                              deps,
                                                              lrn_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx, lrn_index, deps, cpu::mkldnn_utils::OpType::LRN, scratchpad_size);
                    };
                }
                else
                {
                    double alpha = lrn->get_alpha();
                    double beta = lrn->get_beta();
                    double bias = lrn->get_bias();
                    double nsize = lrn->get_nsize();
                    Shape arg_shape = args[0].get_shape();
                    Shape axes_shape = args[1].get_shape();

                    auto element_type = lrn->get_element_type();
                    if (element_type == element::f32)
                    {
                        functor = [&,
                                   alpha,
                                   beta,
                                   bias,
                                   arg_shape,
                                   axes,
                                   axes_shape,
                                   nsize,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            ngraph::runtime::reference::lrn<float>(
                                static_cast<float*>(ctx->buffer_data[arg_buffer_index]),
                                axes,
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                arg_shape,
                                alpha,
                                beta,
                                bias,
                                nsize);
                        };
                    }
                    else if (element_type == element::f64)
                    {
                        functor = [&,
                                   alpha,
                                   beta,
                                   bias,
                                   arg_shape,
                                   axes,
                                   axes_shape,
                                   nsize,
                                   arg_buffer_index,
                                   out_buffer_index](CPURuntimeContext* ctx,
                                                     CPUExecutionContext* /* ectx */) {
                            ngraph::runtime::reference::lrn<double>(
                                static_cast<double*>(ctx->buffer_data[arg_buffer_index]),
                                axes,
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                arg_shape,
                                alpha,
                                beta,
                                bias,
                                nsize);
                        };
                    }
                    else
                    {
                        throw ngraph_error("Unsupported type in CPU Builder for LRN");
                    }
                }

                functors.emplace_back(functor);
            }

            void register_builders_lrn_cpp() { REGISTER_OP_BUILDER(LRN); }
        }
    }
}
