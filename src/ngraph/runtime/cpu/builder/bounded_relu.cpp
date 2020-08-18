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

#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/dnnl_invoke.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/kernel/relu.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::BoundedRelu)
            {
                auto& functors = external_function->get_functors();

                auto input_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t count = out[0].get_size();

                auto alpha = static_cast<const ngraph::op::BoundedRelu*>(node)->get_alpha();

                if (runtime::cpu::dnnl_utils::use_dnnl_kernel(node))
                {
                    auto& dnnl_emitter = external_function->get_dnnl_emitter();
                    auto bounded_relu_desc = dnnl_emitter->get_bounded_relu_desc(node);
                    size_t scratchpad_size = QUERY_SCRATCHPAD(eltwise_forward, bounded_relu_desc);

                    // BoundedRelu needs 3 primitives: input, result, and eltwise_forward.
                    auto bounded_relu_index = dnnl_emitter->reserve_primitive_space(3);
                    auto& deps = dnnl_emitter->get_primitive_deps(bounded_relu_index);

                    auto functor = [&,
                                    bounded_relu_desc,
                                    bounded_relu_index,
                                    scratchpad_size,
                                    input_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            dnnl_emitter->build_bounded_relu(ctx->dnnl_memories,
                                                             ctx->dnnl_primitives,
                                                             ctx->dnnl_scratchpad_mds,
                                                             bounded_relu_desc,
                                                             deps,
                                                             bounded_relu_index);
                        }
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[input_buffer_index]);
                        cpu::dnnl_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::dnnl_utils::dnnl_invoke_primitive(ctx,
                                                               bounded_relu_index,
                                                               deps,
                                                               cpu::dnnl_utils::OpType::BOUNDEDRELU,
                                                               scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::bounded_relu<float>)> kernel;

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
                    functors.emplace_back(functor);
                }
            }

            void register_builders_bounded_relu_cpp()
            {
                REGISTER_OP_BUILDER(ngraph::op::BoundedRelu);
            }
        }
    }
}
