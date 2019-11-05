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
                (void)node;
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());

                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto element_count = args[0].get_size();

                std::function<decltype(runtime::cpu::kernel::select<float>)> kernel;

                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::select)

                auto functor = [&,
                                kernel,
                                element_count,
                                arg0_buffer_index,
                                arg1_buffer_index,
                                arg2_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                    kernel(ctx->buffer_data[arg0_buffer_index],
                           ctx->buffer_data[arg1_buffer_index],
                           ctx->buffer_data[arg2_buffer_index],
                           ctx->buffer_data[out_buffer_index],
                           element_count,
                           ectx->arena);
                };
                functors.emplace_back(functor);
            }

            void register_builders_select_cpp() { REGISTER_OP_BUILDER(Select); }
        }
    }
}
