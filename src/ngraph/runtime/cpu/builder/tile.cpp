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

#include "ngraph/runtime/cpu/kernel/tile.hpp"
#include "ngraph/op/experimental/tile.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Tile)
            {
                auto tile = static_cast<const ngraph::op::Tile*>(node);
                auto arg_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto arg_rank = arg_shape.size();

                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                std::function<decltype(runtime::cpu::kernel::tile<float, 2>)> kernel;
                SELECT_KERNEL_BY_RANK(
                    kernel, out[0].get_element_type(), arg_rank, runtime::cpu::kernel::tile);
                auto functor =
                    [&, kernel, arg_shape, out_shape, arg_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               arg_shape,
                               out_shape,
                               ectx->arena);
                    };

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Tile);
        }
    }
}
