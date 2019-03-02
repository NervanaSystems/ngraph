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

#include "ngraph/op/pad.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/pad.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            CFFunctionTy Builder::BUILDER_CF_DECL(ngraph::op::Pad)
            {
                auto pad = static_cast<const ngraph::op::Pad*>(node);

                auto arg_shape = pad->get_argument(0)->get_shape();
                auto out_shape = pad->get_shape();
                auto padding_below = pad->get_padding_below();
                auto padding_above = pad->get_padding_above();

                if (pad->get_padding_interior() == Shape(arg_shape.size()))
                {
                    std::function<decltype(runtime::cpu::kernel::pad<float, 1>)> kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          pad->get_input_element_type(0),
                                          arg_shape.size(),
                                          runtime::cpu::kernel::pad);

                    auto functor = [kernel, arg_shape, out_shape, padding_below, padding_above](
                        const std::vector<void*> inputs, std::vector<void*> outputs) {
                        kernel(inputs[0],
                               outputs[0],
                               inputs[1],
                               arg_shape,
                               out_shape,
                               padding_below,
                               padding_above,
                               0);
                    };
                    return functor;
                }
                else
                {
                    auto padding_interior = pad->get_padding_interior();

                    std::function<decltype(runtime::cpu::kernel::pad_ref<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, pad->get_input_element_type(0), runtime::cpu::kernel::pad_ref);

                    auto functor = [kernel,
                                    arg_shape,
                                    out_shape,
                                    padding_below,
                                    padding_above,
                                    padding_interior](const std::vector<void*> inputs,
                                                      std::vector<void*> outputs) {
                        kernel(inputs[0],
                               inputs[1],
                               outputs[0],
                               arg_shape,
                               out_shape,
                               padding_below,
                               padding_above,
                               padding_interior,
                               0);
                    };
                    return functor;
                }
            }
            REGISTER_CF_BUILDER(Pad);
        }
    }
}
