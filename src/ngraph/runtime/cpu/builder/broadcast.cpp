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

#include <cstring>

#include "ngraph/op/broadcast.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/broadcast.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            static void get_broadcast_kernel(
                const ngraph::Node* node,
                std::function<decltype(runtime::cpu::kernel::broadcast<float, 2>)>& kernel,
                Shape& expanded_input_shape,
                Shape& out_shape,
                size_t& size)
            {
                auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);
                auto broadcast_axes = broadcast->get_broadcast_axes();

                auto arg_shape = broadcast->get_argument(0)->get_shape();
                out_shape = broadcast->get_shape();

                // TODO(jmenon): Shape transformations, rank reduction etc. needs to be general
                // and not in any one builder. Move this to the Halide analysis phase.

                // Transform output shape - ex. [4, 1, 2, 2] -> [4, 1, 4]
                // if we're not broadcasting along axes 2 and 3

                if (broadcast_axes.size() > 1)
                {
                    auto innermost_axis = broadcast_axes.end();
                    advance(innermost_axis, -1);
                    auto reduced = Shape{};
                    if (broadcast_axes.size() == (*innermost_axis - *broadcast_axes.begin() + 1))
                    {
                        size_t reduced_count = 1;
                        for (auto axis : broadcast_axes)
                        {
                            reduced_count *= out_shape[axis];
                        }

                        bool done = false;
                        for (size_t i = 0; i < out_shape.size(); i++)
                        {
                            if (!broadcast_axes.count(i))
                            {
                                reduced.push_back(out_shape[i]);
                            }
                            else
                            {
                                if (!done)
                                {
                                    reduced.push_back(reduced_count);
                                    done = true;
                                }
                            }
                        }
                        broadcast_axes = AxisSet{*broadcast_axes.begin()};
                        out_shape = reduced;
                    }
                }

                // Squeeze output shape
                // Ex. [2, 1, 1, 2] -> [2, 2]

                auto squeezed_out_shape = Shape{};
                for (size_t i = 0; i < out_shape.size(); i++)
                {
                    if (out_shape[i] != 1)
                    {
                        squeezed_out_shape.push_back(out_shape[i]);
                    }
                    else
                    {
                        broadcast_axes.erase(i);
                        // TODO(jmenon): This needs to be rewritten
                        // when it gets moved to the analysis pass
                        // that doesn't use AxisSet
                        auto new_bcast_axes = AxisSet{};
                        for (auto axis : broadcast_axes)
                        {
                            if (axis > i)
                                new_bcast_axes.insert(axis - 1);
                            else
                                new_bcast_axes.insert(axis);
                        }
                        broadcast_axes = new_bcast_axes;
                    }
                }
                out_shape = squeezed_out_shape;

                // Squeeze input shape
                auto squeezed_arg_shape = Shape{};
                for (size_t i = 0; i < arg_shape.size(); i++)
                {
                    if (arg_shape[i] != 1)
                    {
                        squeezed_arg_shape.push_back(arg_shape[i]);
                    }
                }
                arg_shape = squeezed_arg_shape;

                auto arg_rank = arg_shape.size();
                auto out_rank = out_shape.size();

                if (broadcast_axes.empty())
                {
                    size = shape_size(out_shape) * broadcast->get_element_type().size();
                    return;
                }

                if (!arg_rank)
                {
                    arg_rank = 1;
                    arg_shape = Shape{1};
                }

                // Eigen broadcasts do not reshape their inputs
                // so expand as needed
                // Ex. [2] -> [2, 1] for output shape [2, 4]

                expanded_input_shape = Shape(out_rank, 1);
                size_t i = 0;
                for (size_t j = 0; j < out_rank; j++)
                {
                    if (broadcast_axes.count(j))
                    {
                        expanded_input_shape[j] = 1;
                    }
                    else
                    {
                        expanded_input_shape[j] = arg_shape[i++];
                    }
                }

                SELECT_KERNEL_ET_RANK(kernel,
                                      broadcast->get_input_element_type(0),
                                      out_rank,
                                      runtime::cpu::kernel::broadcast)
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Broadcast)
            {
                std::function<decltype(runtime::cpu::kernel::broadcast<float, 2>)> kernel;
                Shape expanded_input_shape, out_shape;
                size_t size;

                get_broadcast_kernel(node, kernel, expanded_input_shape, out_shape, size);
                NodeExecutorTy functor;
                if (kernel)
                {
                    functor = [kernel, expanded_input_shape, out_shape](
                        const std::vector<void*> inputs, std::vector<void*> outputs) {
                        kernel(inputs[0], outputs[0], expanded_input_shape, out_shape, 0);
                    };
                }
                else
                {
                    functor = [size](const std::vector<void*>& inputs,
                                     std::vector<void*>& outputs) {
                        memcpy(outputs[0], inputs[0], size);
                    };
                }
                return functor;
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::Broadcast)
            {
                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                std::function<decltype(runtime::cpu::kernel::broadcast<float, 2>)> kernel;
                Shape expanded_input_shape, out_shape;
                size_t size;

                get_broadcast_kernel(node, kernel, expanded_input_shape, out_shape, size);
                CPUKernelFunctor functor;
                if (kernel)
                {
                    functor = [&,
                               kernel,
                               expanded_input_shape,
                               out_shape,
                               arg_buffer_index,
                               out_buffer_index](CPURuntimeContext* ctx,
                                                 CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               expanded_input_shape,
                               out_shape,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    functor = [&, size, arg_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                        memcpy(ctx->buffer_data[out_buffer_index],
                               ctx->buffer_data[arg_buffer_index],
                               size);
                    };
                    functors.emplace_back(functor);
                }
            }

            void register_builders_broadcast_cpp()
            {
                REGISTER_CF_BUILDER(Broadcast);
                REGISTER_OP_BUILDER(Broadcast);
            }
        }
    }
}
