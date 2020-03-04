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

#include "ngraph/op/reverse.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/reverse.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Reverse)
            {
                auto reverse = static_cast<const ngraph::op::Reverse*>(node);

                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();
                auto reversed_axes = reverse->get_reversed_axes();

                std::function<decltype(runtime::cpu::kernel::reverse<float>)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::reverse)

                auto functor = [&,
                                kernel,
                                arg_shape,
                                result_shape,
                                reversed_axes,
                                arg_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* /* ectx */) {
                    kernel(ctx->buffer_data[arg_buffer_index],
                           ctx->buffer_data[out_buffer_index],
                           arg_shape,
                           result_shape,
                           reversed_axes);
                };
                functors.emplace_back(functor);
            }

            void register_builders_reverse_cpp() { REGISTER_OP_BUILDER(Reverse); }
        }
    }
}
