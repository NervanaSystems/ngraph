//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto padding_value_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto pad = static_cast<const ngraph::op::Pad*>(node);

                auto arg_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto padding_below = pad->get_padding_below();
                auto padding_above = pad->get_padding_above();
                auto pad_mode = pad->get_pad_mode();

                if ((pad_mode == ngraph::op::PadMode::CONSTANT ||
                     pad_mode == ngraph::op::PadMode::REFLECT) &&
                    is_optimized_et(args[0].get_element_type()))
                {
                    std::function<decltype(runtime::cpu::kernel::pad_and_slice<float, 1>)> kernel;

                    SELECT_ETS_AND_RANK7(kernel,
                                         args[0].get_element_type(),
                                         arg_shape.size(),
                                         runtime::cpu::kernel::pad_and_slice);

                    auto functor = [&,
                                    kernel,
                                    arg_shape,
                                    out_shape,
                                    padding_below,
                                    padding_above,
                                    pad_mode,
                                    arg_buffer_index,
                                    padding_value_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               ctx->buffer_data[padding_value_index],
                               arg_shape,
                               out_shape,
                               CoordinateDiff(padding_below.begin(), padding_below.end()),
                               CoordinateDiff(padding_above.begin(), padding_above.end()),
                               pad_mode,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::pad_ref<float>)> kernel;

                    SELECT_KERNEL(kernel, args[0].get_element_type(), runtime::cpu::kernel::pad_ref)

                    auto functor = [&,
                                    kernel,
                                    arg_shape,
                                    out_shape,
                                    padding_below,
                                    padding_above,
                                    pad_mode,
                                    arg_buffer_index,
                                    padding_value_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[padding_value_index],
                               ctx->buffer_data[out_buffer_index],
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

            template <>
            NodeExecutorTy Builder::BUILDER_CF_DECL(ngraph::op::Pad)
            {
                auto pad = static_cast<const ngraph::op::Pad*>(node);

                auto arg_shape = pad->get_argument(0)->get_shape();
                auto out_shape = pad->get_shape();
                auto padding_below = pad->get_padding_below();
                auto padding_above = pad->get_padding_above();
                auto pad_mode = pad->get_pad_mode();

                if ((pad_mode == ngraph::op::PadMode::CONSTANT ||
                     pad_mode == ngraph::op::PadMode::REFLECT) &&
                    is_optimized_et(pad->get_input_element_type(0)))
                {
                    std::function<decltype(runtime::cpu::kernel::pad_and_slice<float, 1>)> kernel;

                    SELECT_ETS_AND_RANK7(kernel,
                                         pad->get_input_element_type(0),
                                         arg_shape.size(),
                                         runtime::cpu::kernel::pad_and_slice);

                    auto functor =
                        [kernel, arg_shape, out_shape, padding_below, padding_above, pad_mode](
                            const std::vector<void*>& inputs, std::vector<void*>& outputs) {
                            kernel(inputs[0],
                                   outputs[0],
                                   inputs[1],
                                   arg_shape,
                                   out_shape,
                                   CoordinateDiff(padding_below.begin(), padding_below.end()),
                                   CoordinateDiff(padding_above.begin(), padding_above.end()),
                                   pad_mode,
                                   0);
                        };
                    return functor;
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::pad_ref<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, pad->get_input_element_type(0), runtime::cpu::kernel::pad_ref)

                    auto functor =
                        [kernel, arg_shape, out_shape, padding_below, padding_above, pad_mode](
                            const std::vector<void*>& inputs, std::vector<void*>& outputs) {
                            kernel(inputs[0],
                                   inputs[1],
                                   outputs[0],
                                   arg_shape,
                                   out_shape,
                                   padding_below,
                                   padding_above,
                                   pad_mode,
                                   0);
                        };
                    return functor;
                }
            }

            void register_builders_pad_cpp()
            {
                REGISTER_OP_BUILDER(Pad);
                REGISTER_CF_BUILDER(Pad);
            }
        }
    }
}
