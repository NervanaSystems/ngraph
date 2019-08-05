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

#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/softmax.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/reference/softmax.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Softmax)
            {
                auto softmax = static_cast<const ngraph::op::Softmax*>(node);

                auto& functors = external_function->get_functors();

                auto arg_shape = args[0].get_shape();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto axes = softmax->get_axes();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto softmax_desc = mkldnn_emitter->get_softmax_forward_desc(node);
                    // Softmax needs 3 primitives: input, result, and softmax_forward.
                    size_t softmax_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(softmax_index);

                    auto functor =
                        [&, softmax_desc, softmax_index, arg_buffer_index, out_buffer_index](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            if (ctx->first_iteration)
                            {
                                mkldnn_emitter->build_softmax_forward(
                                    ctx->mkldnn_primitives, softmax_desc, deps, softmax_index);
                            }
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[0], ctx->buffer_data[arg_buffer_index]);
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[1], ctx->buffer_data[out_buffer_index]);
                            cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, softmax_index);
                        };
                    functors.emplace_back(functor);
                }
                else
                {
                    if (axes.size() == arg_shape.size())
                    {
                        std::function<decltype(runtime::cpu::kernel::softmax_all<float, 1>)> kernel;

                        PARTIAL_SELECT_KERNEL_BY_RANK(kernel,
                                                      args[0].get_element_type(),
                                                      args[0].get_shape().size(),
                                                      runtime::cpu::kernel::softmax_all);

                        auto functor = [&, kernel, arg_shape, arg_buffer_index, out_buffer_index](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   arg_shape,
                                   ectx->arena);
                        };
                        functors.emplace_back(functor);
                    }
                    else if (axes.size() == 1)
                    {
                        if (*axes.begin() == (arg_shape.size() - 1))
                        {
                            std::function<decltype(
                                runtime::cpu::kernel::softmax_innermost_1rd<float, 1>)>
                                kernel;

                            PARTIAL_SELECT_KERNEL_BY_RANK(
                                kernel,
                                args[0].get_element_type(),
                                args[0].get_shape().size(),
                                runtime::cpu::kernel::softmax_innermost_1rd);

                            auto functor =
                                [&, kernel, arg_shape, arg_buffer_index, out_buffer_index](
                                    CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                                    kernel(ctx->buffer_data[arg_buffer_index],
                                           ctx->buffer_data[out_buffer_index],
                                           arg_shape,
                                           ectx->arena);
                                };
                            functors.emplace_back(functor);
                        }
                        else
                        {
                            std::function<decltype(runtime::cpu::kernel::softmax_1rd<float, 1>)>
                                kernel;

                            PARTIAL_SELECT_KERNEL_BY_RANK(kernel,
                                                          args[0].get_element_type(),
                                                          args[0].get_shape().size(),
                                                          runtime::cpu::kernel::softmax_1rd);

                            auto functor =
                                [&, kernel, arg_shape, axes, arg_buffer_index, out_buffer_index](
                                    CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                                    kernel(ctx->buffer_data[arg_buffer_index],
                                           ctx->buffer_data[out_buffer_index],
                                           arg_shape,
                                           axes,
                                           ectx->arena);
                                };
                            functors.emplace_back(functor);
                        }
                    }
                    else if (arg_shape.size() == 3 && axes.size() == 2)
                    {
                        std::function<decltype(runtime::cpu::kernel::softmax_3d_2rd<float>)> kernel;

                        SELECT_KERNEL(kernel,
                                      args[0].get_element_type(),
                                      runtime::cpu::kernel::softmax_3d_2rd);

                        auto functor =
                            [&, kernel, arg_shape, axes, arg_buffer_index, out_buffer_index](
                                CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                                kernel(ctx->buffer_data[arg_buffer_index],
                                       ctx->buffer_data[out_buffer_index],
                                       arg_shape,
                                       axes,
                                       ectx->arena);
                            };
                        functors.emplace_back(functor);
                    }
                    else if (arg_shape.size() == 4 && axes.size() == 3)
                    {
                        std::function<decltype(runtime::cpu::kernel::softmax_4d_3rd<float>)> kernel;

                        SELECT_KERNEL(kernel,
                                      args[0].get_element_type(),
                                      runtime::cpu::kernel::softmax_4d_3rd);

                        auto functor =
                            [&, kernel, arg_shape, axes, arg_buffer_index, out_buffer_index](
                                CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                                kernel(ctx->buffer_data[arg_buffer_index],
                                       ctx->buffer_data[out_buffer_index],
                                       arg_shape,
                                       axes,
                                       ectx->arena);
                            };
                        functors.emplace_back(functor);
                    }
                    else if (softmax->get_element_type() == element::f32)
                    {
                        NGRAPH_WARN << "Falling back to refernce kernel for softmax " << arg_shape
                                    << " over " << axes;
                        auto functor = [&, arg_shape, axes, arg_buffer_index, out_buffer_index](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            runtime::reference::softmax<float>(
                                static_cast<float*>(ctx->buffer_data[arg_buffer_index]),
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                arg_shape,
                                axes);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        NGRAPH_ERR << "Unsupported Softmax " << arg_shape << " over " << axes
                                   << " in cpu buiilder";
                        throw ngraph_error("Unsupported Softmax");
                    }
                }
            }

            REGISTER_OP_BUILDER(Softmax);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_softmax_cpp() {}
#endif
        }
    }
}
