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

#include "ngraph/runtime/cpu/kernel/reduce_function.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/tensor_view.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Reduce)
            {
                auto reduce = static_cast<const ngraph::op::Reduce*>(node);
                auto function = reduce->get_functions()[0];

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

                auto reduction_axes = reduce->get_reduction_axes();

                if (reduction_axes.empty())
                {
                    size_t size = args[0].get_size() * args[0].get_element_type().size();
                    auto functor = [&, size](CPURuntimeContext* ctx) {
                        memcpy(out_tensor, arg0_tensor, size);
                    };
                    functors.emplace_back(functor);
                }
                else if (reduction_axes.size() == 1)
                {
                    std::function<decltype(runtime::cpu::kernel::reduce_function_1rd<float, 1>)>
                        kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          args[0].get_element_type(),
                                          arg0_shape.size(),
                                          runtime::cpu::kernel::reduce_function_1rd);

                    auto functor =
                        [&, kernel, arg0_shape, out_shape, reduction_axes](CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   out_shape,
                                   reduction_axes,
                                   reducer_external_function);
                        };
                    functors.emplace_back(functor);
                }
                else if (arg0_shape.size() == 2 && reduction_axes.size() == 2)
                {
                    std::function<decltype(runtime::cpu::kernel::reduce_function_2d_2rd<float>)>
                        kernel;

                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::reduce_function_2d_2rd);

                    auto functor =
                        [&, kernel, arg0_shape, out_shape, reduction_axes](CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   out_shape,
                                   reduction_axes,
                                   reducer_external_function);
                        };
                    functors.emplace_back(functor);
                }
                else if (arg0_shape.size() == 3 && reduction_axes.size() == 2)
                {
                    std::function<decltype(runtime::cpu::kernel::reduce_function_3d_2rd<float>)>
                        kernel;

                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::reduce_function_3d_2rd);

                    auto functor =
                        [&, kernel, arg0_shape, out_shape, reduction_axes](CPURuntimeContext* ctx) {
                            kernel(arg0_tensor,
                                   arg1_tensor,
                                   out_tensor,
                                   arg0_shape,
                                   out_shape,
                                   reduction_axes,
                                   reducer_external_function);
                        };
                    functors.emplace_back(functor);
                }
                else
                {
                    throw ngraph_error("Unsupported Reduce");
                }
            }

            REGISTER_OP_BUILDER(Reduce);
        }
    }
}
