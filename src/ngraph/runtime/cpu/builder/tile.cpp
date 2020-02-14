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
                (void)node;
                auto arg_shape = args[0].get_shape();
                auto arg_rank = arg_shape.size();

                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto out_shape = out[0].get_shape();

                // keep it here in case we want to support scalar input in the future.
                if (arg_rank == 0)
                {
                    size_t repeats = shape_size(out_shape);
                    std::function<decltype(runtime::cpu::kernel::tile_rank_0<float>)> kernel;
                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::tile_rank_0)
                    auto functor = [&, kernel, repeats, arg_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                        kernel(ctx->buffer_data[arg_buffer_index],
                               ctx->buffer_data[out_buffer_index],
                               repeats);
                    };

                    functors.emplace_back(functor);
                }
                else
                {
                    auto out_rank = out_shape.size();
                    arg_shape.insert(arg_shape.begin(), out_rank - arg_rank, 1);
                    std::function<decltype(runtime::cpu::kernel::tile<float, 2>)> kernel;
                    SELECT_KERNEL_ET_RANK(
                        kernel, out[0].get_element_type(), out_rank, runtime::cpu::kernel::tile);
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
            }

            void register_builders_tile_cpp() { REGISTER_OP_BUILDER(Tile); }
        }
    }
}
