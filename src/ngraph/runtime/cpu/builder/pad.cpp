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
#include "ngraph/runtime/cpu/kernel/slice.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Pad)
            {
                using namespace std::placeholders;

                auto& functors = external_function->get_functors();

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& padding_value = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto pad = static_cast<const ngraph::op::Pad*>(node);

                auto arg_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto padding_below = pad->get_padding_below();
                auto padding_above = pad->get_padding_above();
                auto pad_mode = pad->get_pad_mode();

                if (pad->get_pad_mode() == ngraph::op::PadMode::CONSTANT)
                {
                    std::function<decltype(runtime::cpu::kernel::pad_and_slice<float, 1>)> kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          args[0].get_element_type(),
                                          arg_shape.size(),
                                          runtime::cpu::kernel::pad_and_slice);

                    auto functor = [&, kernel, arg_shape, out_shape, padding_below, padding_above](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(arg_tensor,
                               out_tensor,
                               padding_value,
                               arg_shape,
                               out_shape,
                               CoordinateDiff(padding_below.begin(), padding_below.end()),
                               CoordinateDiff(padding_above.begin(), padding_above.end()),
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::pad_ref<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::pad_ref);

                    auto functor =
                        [&, kernel, arg_shape, out_shape, padding_below, padding_above, pad_mode](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            kernel(arg_tensor,
                                   padding_value,
                                   out_tensor,
                                   arg_shape,
                                   out_shape,
                                   padding_below,
                                   padding_above,
                                   pad_mode,
                                   ectx->arena);
                        };
                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(Pad);
        }
    }
}
