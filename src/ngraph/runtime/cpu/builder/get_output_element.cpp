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

#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::GetOutputElement)
            {
                (void)node;
                auto& functors = external_function->get_functors();
                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto functor = [&, arg_buffer_index, out_buffer_index](
                    CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                    if (ctx->buffer_data[arg_buffer_index] != ctx->buffer_data[out_buffer_index])
                    {
                        throw ngraph_error("GOE's input and out must be equal");
                    }
                };
                functors.emplace_back(functor);
                return;
            }

            void register_builders_get_output_element_cpp()
            {
                REGISTER_OP_BUILDER(GetOutputElement);
            }
        }
    }
}
