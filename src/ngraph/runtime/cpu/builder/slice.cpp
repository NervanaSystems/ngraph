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

#include "ngraph/op/slice.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/slice.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Slice)
            {
                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                const ngraph::op::Slice* slice = static_cast<const ngraph::op::Slice*>(node);

                auto arg_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();

                auto strides = slice->get_strides();
                auto lower_bounds = slice->get_lower_bounds();
                auto upper_bounds = slice->get_upper_bounds();

                if (auto op_annotations = slice->get_op_annotations())
                {
                    auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                    if (in_place_oi_pairs.size() > 0)
                    {
                        auto element_size = slice->get_input_element_type(0).size();
                        auto start = 0, accumulated = 1;
                        for (int i = arg_shape.size() - 1; i >= 0; i--)
                        {
                            start += lower_bounds[i] * accumulated;
                            accumulated *= arg_shape[i];
                        }
                        auto out_size = shape_size(out_shape) * element_size;
                        auto arg_size = shape_size(arg_shape) * element_size;
                        auto offset = start * element_size;

                        auto functor = [&,
                                        out_size,
                                        arg_size,
                                        offset,
                                        arg_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* /* ectx */) {
                            if (ctx->buffer_data[out_buffer_index] <
                                    ctx->buffer_data[arg_buffer_index] ||
                                ctx->buffer_data[out_buffer_index] >=
                                    reinterpret_cast<char*>(ctx->buffer_data[arg_buffer_index]) +
                                        arg_size)
                            {
                                memcpy(ctx->buffer_data[out_buffer_index],
                                       reinterpret_cast<char*>(ctx->buffer_data[arg_buffer_index]) +
                                           offset,
                                       out_size);
                            }
                        };
                        functors.emplace_back(functor);
                        return;
                    }
                }

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    QUERY_SCRATCHPAD_4ARGS(slice, input_desc, result_desc, lower_bounds, out_shape);

                    // Slice needs 3 primitives: input, result, and reorder.
                    auto slice_index = mkldnn_emitter->reserve_primitive_space(3);
                    auto& deps = mkldnn_emitter->get_primitive_deps(slice_index);

                    auto functor = [&,
                                    input_desc,
                                    result_desc,
                                    lower_bounds,
                                    out_shape,
                                    slice_index,
                                    arg_buffer_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_slice(ctx->mkldnn_memories,
                                                        ctx->mkldnn_primitives,
                                                        ctx->mkldnn_scratchpad_mds,
                                                        input_desc,
                                                        result_desc,
                                                        lower_bounds,
                                                        out_shape,
                                                        deps,
                                                        slice_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx, slice_index, deps, cpu::mkldnn_utils::OpType::SLICE);
                    };

                    functors.emplace_back(functor);
                }
                else
                {
                    if (is_strided(strides) && is_optimized_et(args[0].get_element_type()))
                    {
                        std::function<decltype(runtime::cpu::kernel::strided_slice<float, 2>)>
                            kernel;

                        SELECT_ETS_AND_RANK7(kernel,
                                             args[0].get_element_type(),
                                             arg_shape.size(),
                                             runtime::cpu::kernel::strided_slice);

                        auto functor = [&,
                                        kernel,
                                        arg_shape,
                                        out_shape,
                                        lower_bounds,
                                        upper_bounds,
                                        strides,
                                        arg_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   arg_shape,
                                   out_shape,
                                   lower_bounds,
                                   upper_bounds,
                                   strides,
                                   ectx->arena);
                        };
                        functors.emplace_back(functor);
                    }
                    else if (is_optimized_et(args[0].get_element_type()))
                    {
                        std::function<decltype(runtime::cpu::kernel::slice<float, 2>)> kernel;

                        SELECT_ETS_AND_RANK7(kernel,
                                             args[0].get_element_type(),
                                             arg_shape.size(),
                                             runtime::cpu::kernel::slice);

                        auto functor = [&,
                                        kernel,
                                        arg_shape,
                                        out_shape,
                                        lower_bounds,
                                        arg_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* ectx) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   arg_shape,
                                   out_shape,
                                   lower_bounds,
                                   ectx->arena);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        std::function<decltype(runtime::cpu::kernel::ref_slice<float>)> kernel;
                        SELECT_KERNEL(
                            kernel, args[0].get_element_type(), runtime::cpu::kernel::ref_slice);
                        auto functor = [&,
                                        kernel,
                                        arg_shape,
                                        out_shape,
                                        lower_bounds,
                                        upper_bounds,
                                        strides,
                                        arg_buffer_index,
                                        out_buffer_index](CPURuntimeContext* ctx,
                                                          CPUExecutionContext* /*ectx*/) {
                            kernel(ctx->buffer_data[arg_buffer_index],
                                   ctx->buffer_data[out_buffer_index],
                                   arg_shape,
                                   lower_bounds,
                                   upper_bounds,
                                   strides,
                                   out_shape);
                        };
                        functors.emplace_back(functor);
                    }
                }
            }

            void register_builders_slice_cpp() { REGISTER_OP_BUILDER(Slice); }
        }
    }
}
