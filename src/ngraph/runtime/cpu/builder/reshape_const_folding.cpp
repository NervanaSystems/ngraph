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
#include "ngraph/util.hpp"
//#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
//#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            CFFunctionTy Builder::BUILDER_CF_DECL(ngraph::op::Reshape)
            {
                auto reshape = static_cast<const ngraph::op::Reshape*>(node);

                auto arg_shape = reshape->get_argument(0)->get_shape();
                auto arg_rank = arg_shape.size();

                auto result_shape = reshape->get_output_shape();
                auto result_rank = result_shape.size();
                auto& result_element_type = reshape->get_element_type();

                auto input_order = reshape->get_input_order();

                bool same_layout = is_sorted(input_order.begin(), input_order.end());

                auto result_size = shape_size(result_shape);
                size_t size = result_size * result_element_type.size();

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
                    auto functor = [size](std::vector<void*> inputs, std::vector<void*> outputs) {
                        if (inputs[0] != outputs[0])
                        {
                            memcpy(outputs[0], inputs[0], size);
                        }
                    };
                    return functor;
                }

                if (same_layout || result_size < 2)
                {
                    auto functor = [size](std::vector<void*> inputs, std::vector<void*> outputs) {
                        memcpy(outputs[0], inputs[0], size);
                    };
                    return functor;
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
                    std::function<decltype(runtime::cpu::kernel::reshape_ref<float>)> ref_kernel;

                    SELECT_KERNEL(
                        ref_kernel, result_element_type, runtime::cpu::kernel::reshape_ref);

                    auto functor = [ref_kernel, arg_shape, input_order, result_shape](
                        std::vector<void*> inputs, std::vector<void*> outputs) {
                        ref_kernel(inputs[0], outputs[0], arg_shape, input_order, result_shape, 0);
                    };
                    return functor;
                }

                auto functor = [kernel, arg_shape, input_order, result_shape](
                    std::vector<void*> inputs, std::vector<void*> outputs) {
                    kernel(inputs[0], outputs[0], arg_shape, input_order, result_shape, 0);
                };
                return functor;
            }
            REGISTER_CF_BUILDER(Reshape);
        }
    }
}
