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

#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/reference/all.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/tensor.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Any)
            {
                auto& functors = external_function->get_functors();
                auto reduce = static_cast<const ngraph::op::Any*>(node);
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto reduction_axes = reduce->get_reduction_axes();
                auto functor =
                    [&, arg0_shape, out_shape, reduction_axes, arg0_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::reference::any(
                            static_cast<char*>(ctx->buffer_data[arg0_buffer_index]),
                            static_cast<char*>(ctx->buffer_data[out_buffer_index]),
                            arg0_shape,
                            out_shape,
                            reduction_axes);
                    };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::All)
            {
                auto& functors = external_function->get_functors();
                auto reduce = static_cast<const ngraph::op::All*>(node);
                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto arg0_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto reduction_axes = reduce->get_reduction_axes();
                auto functor =
                    [&, arg0_shape, out_shape, reduction_axes, arg0_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        runtime::reference::all(
                            static_cast<char*>(ctx->buffer_data[arg0_buffer_index]),
                            static_cast<char*>(ctx->buffer_data[out_buffer_index]),
                            arg0_shape,
                            out_shape,
                            reduction_axes);
                    };
                functors.emplace_back(functor);
            }

            void register_builders_reduce_function_cpp()
            {
                REGISTER_OP_BUILDER(Any);
                REGISTER_OP_BUILDER(All);
            }
        }
    }
}
