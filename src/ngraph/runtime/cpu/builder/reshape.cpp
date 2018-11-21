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
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Reshape)
            {
                auto& functors = external_function->get_functors();

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto reshape = static_cast<const ngraph::op::Reshape*>(node);

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
                    size_t size = out[0].get_size() * out[0].get_element_type().size();
                    auto functor = [&, size](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        if (out_tensor != arg_tensor)
                        {
                            memcpy(out_tensor, arg_tensor, size);
                        }
                    };
                    functors.emplace_back(functor);
                    return;
                }

                auto arg_shape = args[0].get_shape();
                auto arg_rank = arg_shape.size();

                auto result_shape = out[0].get_shape();
                auto result_rank = result_shape.size();
                auto& result_element_type = out[0].get_element_type();

                auto input_order = reshape->get_input_order();

                bool same_layout = is_sorted(input_order.begin(), input_order.end());

                auto result_size = shape_size(result_shape);

                if (same_layout || result_size < 2)
                {
                    size_t size = out[0].get_size() * out[0].get_element_type().size();
                    auto functor = [&, size](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        memcpy(out_tensor, arg_tensor, size);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                std::function<decltype(runtime::cpu::kernel::reshape_1d<float, 2>)> kernel;
                if (arg_rank == 1)
                {
                    SELECT_KERNEL_BY_RANK(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_1d);
                }
                else if (arg_rank == 2)
                {
                    SELECT_KERNEL_BY_RANK(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_2d);
                }
                else if (arg_rank == 3)
                {
                    SELECT_KERNEL_BY_RANK(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_3d);
                }
                else if (arg_rank == 4)
                {
                    SELECT_KERNEL_BY_RANK(
                        kernel, result_element_type, result_rank, runtime::cpu::kernel::reshape_4d);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::reshape<float>)> ref_kernel;

                    SELECT_KERNEL(ref_kernel, result_element_type, runtime::cpu::kernel::reshape);

                    auto functor = [&, ref_kernel, arg_shape, input_order, result_shape](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        ref_kernel(arg_tensor,
                                   out_tensor,
                                   arg_shape,
                                   input_order,
                                   result_shape,
                                   ectx->arena);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                auto functor = [&, kernel, arg_shape, input_order, result_shape](
                    CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                    kernel(
                        arg_tensor, out_tensor, arg_shape, input_order, result_shape, ectx->arena);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Reshape);
        }
    }
}
