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

#include "ngraph/runtime/cpu/kernel/reduce_function_window.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/tensor.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::ReduceWindow)
            {
                auto reduce_window = static_cast<const ngraph::op::ReduceWindow*>(node);
                auto function = reduce_window->get_functions()[0];

                auto& functors = external_function->get_functors();
                auto& callees = external_function->get_callees();

                if (!callees.count(function->get_name()))
                {
                    callees[function->get_name()] = make_shared<CPU_ExternalFunction>(function);
                }
                auto& reducer_external_function = callees[function->get_name()];

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto window_shape = reduce_window->get_window_shape();
                auto window_movement_strides = reduce_window->get_window_movement_strides();

                std::function<decltype(runtime::cpu::kernel::reduce_function_window<float>)> kernel;

                SELECT_KERNEL(kernel,
                              args[0].get_element_type(),
                              runtime::cpu::kernel::reduce_function_window);

                auto functor =
                    [&, kernel, arg0_shape, out_shape, window_shape, window_movement_strides](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(arg0_tensor,
                               arg1_tensor,
                               out_tensor,
                               arg0_shape,
                               out_shape,
                               window_shape,
                               window_movement_strides,
                               reducer_external_function);
                    };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(ReduceWindow);
        }
    }
}
