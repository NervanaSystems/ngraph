/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

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
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Broadcast)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto arg_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto arg_rank = arg_shape.size();
                auto out_rank = out_shape.size();

                auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);

                if (broadcast->get_broadcast_axes().empty())
                {
                    size_t size = out[0].get_size() * out[0].get_element_type().size();
                    auto functor = [&, size](CPURuntimeContext* ctx) {
                        memcpy(out_tensor, arg_tensor, size);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if (!arg_rank)
                {
                    arg_rank = 1;
                    arg_shape = Shape{1};
                }
                auto new_shape = Shape(out_rank, 1);
                const auto& broadcast_axes = broadcast->get_broadcast_axes();
                size_t i = 0;
                for (size_t j = 0; j < out_rank; j++)
                {
                    if (broadcast_axes.count(j))
                    {
                        new_shape[j] = 1;
                    }
                    else
                    {
                        new_shape[j] = arg_shape[i++];
                    }
                }

                std::function<decltype(runtime::cpu::kernel::broadcast<float, 2>)> kernel;

                SELECT_KERNEL_BY_RANK(
                    kernel, args[0].get_element_type(), out_rank, runtime::cpu::kernel::broadcast);

                auto functor = [&, kernel, new_shape, out_shape](CPURuntimeContext* ctx) {
                    kernel(arg_tensor, out_tensor, new_shape, out_shape);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Broadcast);
        }
    }
}
