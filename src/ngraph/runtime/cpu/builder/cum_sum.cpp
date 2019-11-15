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

#include "ngraph/runtime/cpu/kernel/cum_sum.hpp"
#include "ngraph/op/cum_sum.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::CumSum)
            {
                (void)node;
                auto cumsum_op = static_cast<const ngraph::op::CumSum*>(node);
                auto element_type = args[0].get_element_type();
                auto in_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto& functors = external_function->get_functors();

                std::function<decltype(runtime::cpu::kernel::reference_cumsum<float>)> kernel;
                SELECT_KERNEL(
                    kernel, args[0].get_element_type(), runtime::cpu::kernel::reference_cumsum)
                auto functor = [&,
                                kernel,
                                arg0_buffer_index,
                                out0_buffer_index,
                                in_shape,
                                out_shape,
                                cumsum_op](CPURuntimeContext* ctx,
                                           CPUExecutionContext* /* ectx */) {
                    kernel(ctx->buffer_data[arg0_buffer_index],
                           ctx->buffer_data[out0_buffer_index],
                           in_shape,
                           out_shape,
                           cumsum_op->get_axis(),
                           cumsum_op->is_exclusive(),
                           cumsum_op->is_reverse());
                };

                functors.emplace_back(functor);
            }

            void register_builders_cumsum_cpp() { REGISTER_OP_BUILDER(CumSum); }
        }
    }
}
