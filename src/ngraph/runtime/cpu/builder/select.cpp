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

#include "ngraph/op/select.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/select.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Select)
            {
                auto& functors = external_function->get_functors();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());

                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto element_count = args[0].get_size();

                std::function<decltype(runtime::cpu::kernel::select<float>)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::select);

                auto functor = [&, kernel, element_count](CPURuntimeContext* ctx) {
                    kernel(arg0_tensor, arg1_tensor, arg2_tensor, out_tensor, element_count);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Select);
        }
    }
}
