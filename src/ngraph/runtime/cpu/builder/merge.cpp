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

#include "ngraph/op/merge.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/merge.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Merge)
            {
                auto& functors = external_function->get_functors();

                auto cond_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto true_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto false_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto count = shape_size(out[0].get_shape());

                std::function<decltype(runtime::cpu::kernel::merge<float>)> kernel;
                SELECT_KERNEL(kernel, out[0].get_element_type(), runtime::cpu::kernel::merge);
                auto functor = [&,
                                kernel,
                                count,
                                cond_buffer_index,
                                true_buffer_index,
                                false_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                    kernel(ctx->buffer_data[cond_buffer_index],
                           ctx->buffer_data[true_buffer_index],
                           ctx->buffer_data[false_buffer_index],
                           ctx->buffer_data[out_buffer_index],
                           count,
                           ectx->arena);
                };

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Merge);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_merge_cpp() {}
#endif
        }
    }
}
