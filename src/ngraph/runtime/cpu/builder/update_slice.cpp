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

#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/update_slice.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::UpdateSlice)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());

                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto update_slice = static_cast<const ngraph::op::UpdateSlice*>(node);

                auto arg0_shape = args[0].get_shape();
                auto arg1_shape = args[1].get_shape();

                auto strides = update_slice->get_strides();
                auto lower_bounds = update_slice->get_lower_bounds();
                auto upper_bounds = update_slice->get_upper_bounds();

                if (!arg0_shape.size())
                {
                    size_t size = args[0].get_element_type().size();
                    auto functor = [&, size, arg1_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        memcpy(ctx->buffer_data[out_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               size);
                    };
                    functors.emplace_back(functor);
                    return;
                }

                if (ngraph::is_strided(strides))
                {
                    std::function<decltype(runtime::cpu::kernel::strided_update_slice<float, 2>)>
                        kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          args[0].get_element_type(),
                                          arg0_shape.size(),
                                          runtime::cpu::kernel::strided_update_slice);

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    lower_bounds,
                                    upper_bounds,
                                    strides,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg0_shape,
                               arg1_shape,
                               lower_bounds,
                               upper_bounds,
                               strides,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::update_slice<float, 2>)> kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          args[0].get_element_type(),
                                          arg0_shape.size(),
                                          runtime::cpu::kernel::update_slice);

                    auto functor = [&,
                                    kernel,
                                    arg0_shape,
                                    arg1_shape,
                                    lower_bounds,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg0_shape,
                               arg1_shape,
                               lower_bounds,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
            }

            void register_builders_update_slice_cpp() { REGISTER_OP_BUILDER(UpdateSlice); }
        }
    }
}
