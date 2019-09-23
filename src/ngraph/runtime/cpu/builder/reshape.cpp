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

#include <cstring>

#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/reshape.hpp"
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
            static void get_reshape_kernel(
                const ngraph::Node* node,
                std::function<decltype(runtime::cpu::kernel::reshape_1d<float, 2>)>& kernel,
                std::function<decltype(runtime::cpu::kernel::reshape_ref<float>)>& ref_kernel,
                Shape& arg_shape,
                Shape& result_shape,
                AxisVector& input_order,
                size_t& size,
                bool& skip_reshape)
            {
                auto reshape = static_cast<const ngraph::op::Reshape*>(node);

                arg_shape = reshape->get_argument(0)->get_shape();
                auto arg_rank = arg_shape.size();

                result_shape = reshape->get_output_shape();
                auto result_rank = result_shape.size();
                auto& result_element_type = reshape->get_element_type();

                input_order = reshape->get_input_order();

                bool same_layout = is_sorted(input_order.begin(), input_order.end());

                auto result_size = shape_size(result_shape);
                size = result_size * result_element_type.size();

                auto can_skip_reshape = [&]() {
                    if (!reshape->get_is_transpose())
                    {
                        return true;
                    }
                    auto annotation = reshape->get_op_annotations();
                    if (annotation && annotation->get_in_place_oi_pairs().size() > 0)
                    {
                        return true;
                    }
                    return false;
                };

                if (can_skip_reshape())
                {
                    skip_reshape = true;
                    return;
                }

                if (same_layout || result_size < 2)
                {
                    return;
                }

                if (arg_rank == 1 && is_optimized_et(result_element_type))
                {
                    SELECT_ETS_AND_RANK7(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_1d);
                }
                else if (arg_rank == 2 && is_optimized_et(result_element_type))
                {
                    SELECT_ETS_AND_RANK7(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_2d);
                }
                else if (arg_rank == 3 && is_optimized_et(result_element_type))
                {
                    SELECT_ETS_AND_RANK7(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_3d);
                }
                else if (arg_rank == 4 && is_optimized_et(result_element_type))
                {
                    SELECT_ETS_AND_RANK7(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_4d);
                }
                else
                {
                    SELECT_KERNEL(
                        ref_kernel, result_element_type, runtime::cpu::kernel::reshape_ref)
                }
            }

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Reshape)
            {
                std::function<decltype(runtime::cpu::kernel::reshape_1d<float, 2>)> kernel;
                std::function<decltype(runtime::cpu::kernel::reshape_ref<float>)> ref_kernel;
                Shape arg_shape, result_shape;
                AxisVector input_order;
                size_t size;
                bool skip_reshape = false;

                get_reshape_kernel(node,
                                   kernel,
                                   ref_kernel,
                                   arg_shape,
                                   result_shape,
                                   input_order,
                                   size,
                                   skip_reshape);
                NodeExecutorTy functor;
                if (kernel)
                {
                    functor = [kernel, arg_shape, input_order, result_shape](
                        const std::vector<void*>& inputs, std::vector<void*>& outputs) {
                        kernel(inputs[0], outputs[0], arg_shape, input_order, result_shape, 0);
                    };
                }
                else if (ref_kernel)
                {
                    functor = [ref_kernel, arg_shape, input_order, result_shape](
                        std::vector<void*> inputs, std::vector<void*> outputs) {
                        ref_kernel(inputs[0], outputs[0], arg_shape, input_order, result_shape, 0);
                    };
                }
                else if (skip_reshape)
                {
                    functor = [size](const std::vector<void*>& inputs,
                                     std::vector<void*>& outputs) {
                        if (inputs[0] != outputs[0])
                        {
                            memcpy(outputs[0], inputs[0], size);
                        }
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
            void Builder::BUILDER_DECL(ngraph::op::Reshape)
            {
                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                std::function<decltype(runtime::cpu::kernel::reshape_1d<float, 2>)> kernel;
                std::function<decltype(runtime::cpu::kernel::reshape_ref<float>)> ref_kernel;
                Shape arg_shape, result_shape;
                AxisVector input_order;
                size_t size;
                bool skip_reshape = false;

                get_reshape_kernel(node,
                                   kernel,
                                   ref_kernel,
                                   arg_shape,
                                   result_shape,
                                   input_order,
                                   size,
                                   skip_reshape);
                CPUKernelFunctor functor;
                if (kernel)
                {
                    functor = [&,
                               kernel,
                               arg_shape,
                               input_order,
                               result_shape,
                               arg_buffer_index,
                               out_buffer_index](CPURuntimeContext* ctx,
                                                 CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg_shape,
                               input_order,
                               result_shape,
                               ectx->arena);
                    };
                }
                else if (ref_kernel)
                {
                    functor = [&,
                               ref_kernel,
                               arg_shape,
                               input_order,
                               result_shape,
                               arg_buffer_index,
                               out_buffer_index](CPURuntimeContext* ctx,
                                                 CPUExecutionContext* ectx) {
                        ref_kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   arg_shape,
                                   input_order,
                                   result_shape,
                                   ectx->arena);
                    };
                }
                else if (skip_reshape)
                {
                    functor = [&, size, arg_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                        if (ctx->buffer_data[out_buffer_index] !=
                            ctx->buffer_data[arg_buffer_index])
                        {
                            memcpy(ctx->buffer_data[out_buffer_index],
                                   ctx->buffer_data[arg_buffer_index],
                                   size);
                        }
                    };
                }
                else
                {
                    functor = [&, size, arg_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                        memcpy(ctx->buffer_data[out_buffer_index],
                               ctx->buffer_data[arg_buffer_index],
                               size);
                    };
                }
                functors.emplace_back(functor);
            }

            void register_builders_reshape_cpp()
            {
                REGISTER_CF_BUILDER(Reshape);
                REGISTER_OP_BUILDER(Reshape);
            }
        }
    }
}
