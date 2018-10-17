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

#include "ngraph/op/convert.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/convert.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Convert)
            {
                auto& functors = external_function->get_functors();

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto element_count = out[0].get_size();

                std::function<decltype(runtime::cpu::kernel::convert<float, int>)> kernel;

                if (out[0].get_element_type() == element::boolean)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_i8);
                }
                else if (out[0].get_element_type() == element::f32)
                {
                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::convert_to_float32);
                }
                else if (out[0].get_element_type() == element::f64)
                {
                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::convert_to_float64);
                }
                else if (out[0].get_element_type() == element::i8)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_i8);
                }
                else if (out[0].get_element_type() == element::i16)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_i16);
                }
                else if (out[0].get_element_type() == element::i32)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_i32);
                }
                else if (out[0].get_element_type() == element::i64)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_i64);
                }
                else if (out[0].get_element_type() == element::u8)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_u8);
                }
                else if (out[0].get_element_type() == element::u16)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_u16);
                }
                else if (out[0].get_element_type() == element::u32)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_u32);
                }
                else if (out[0].get_element_type() == element::u64)
                {
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::convert_to_u64);
                }
                else
                {
                    throw ngraph_error("Cannot convert from an invalid input element type");
                }

                auto functor = [&, kernel, element_count](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* ectx) {
                    kernel(arg_tensor, out_tensor, element_count, ectx->arena);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Convert);
        }
    }
}
