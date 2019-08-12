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

#include "ngraph/runtime/cpu/kernel/erf.hpp"
#include "ngraph/op/erf.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Erf)
            {
                auto element_type = args[0].get_element_type();
                auto element_count = out[0].get_size();
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto& functors = external_function->get_functors();

                if (element_type == element::f32 || element_type == element::f64)
                {
                    std::function<decltype(runtime::cpu::kernel::erf<float>)> kernel;
                    if (element_type == element::f32)
                    {
                        kernel = runtime::cpu::kernel::erf<float>;
                    }
                    else if (element_type == element::f64)
                    {
                        kernel = runtime::cpu::kernel::erf<double>;
                    }
                    auto functor = [&, kernel, element_count, arg0_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               element_count,
                               ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::reference_erf<float>)> kernel;
                    SELECT_KERNEL(
                        kernel, args[0].get_element_type(), runtime::cpu::kernel::reference_erf);
                    auto functor = [&, kernel, element_count, arg0_buffer_index, out0_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               element_count);
                    };

                    functors.emplace_back(functor);
                }
            }
            REGISTER_OP_BUILDER(Erf);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_erf_cpp() {}
#endif
        }
    }
}
