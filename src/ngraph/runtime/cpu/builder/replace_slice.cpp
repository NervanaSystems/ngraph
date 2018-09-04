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

#include "ngraph/op/replace_slice.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/replace_slice.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::ReplaceSlice)
            {
                auto& functors = external_function->get_functors();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());

                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto replace_slice = static_cast<const ngraph::op::ReplaceSlice*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();

                auto strides = replace_slice->get_strides();
                auto lower_bounds = replace_slice->get_lower_bounds();
                auto upper_bounds = replace_slice->get_upper_bounds();

                bool strided = false;
                for (auto stride : strides)
                {
                    if (stride != 1)
                    {
                        strided = true;
                        break;
                    }
                }

                if (!arg0_shape.size())
                {
                    size_t size = args[0].get_element_type().size();
                    auto functor = [&, size](CPURuntimeContext* ctx) {
                        memcpy(out_tensor, arg1_tensor, size);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if (strided)
                {
                    std::function<decltype(runtime::cpu::kernel::strided_replace_slice<float, 2>)>
                        kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          args[0].get_element_type(),
                                          arg0_shape.size(),
                                          runtime::cpu::kernel::strided_replace_slice);

                    auto functor =
                        [&, kernel, arg0_shape, arg1_shape, lower_bounds, upper_bounds, strides](
                            CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   arg1_shape,
                                   lower_bounds,
                                   upper_bounds,
                                   strides);
                        };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::replace_slice<float, 2>)> kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          args[0].get_element_type(),
                                          arg0_shape.size(),
                                          runtime::cpu::kernel::replace_slice);

                    auto functor =
                        [&, kernel, arg0_shape, arg1_shape, lower_bounds](CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   arg1_shape,
                                   lower_bounds);
                        };
                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(ReplaceSlice);
        }
    }
}
