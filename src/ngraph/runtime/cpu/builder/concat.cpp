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

#include "ngraph/op/concat.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/concat.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Concat)
            {
                auto concat = static_cast<const ngraph::op::Concat*>(node);
                auto axis = concat->get_concatenation_axis();

                auto& functors = external_function->get_functors();

                vector<size_t> arg_buffer_indices;
                vector<Shape> arg_shapes;
                vector<size_t> arg_sizes;
                auto element_size = concat->get_input_element_type(0).size();
                for (auto& arg : args)
                {
                    if (shape_size(arg.get_shape()))
                    {
                        arg_buffer_indices.emplace_back(
                            external_function->get_buffer_index(arg.get_name()));
                        arg_shapes.emplace_back(arg.get_shape());
                        arg_sizes.emplace_back(shape_size(arg.get_shape()) * element_size);
                    }
                }
                auto nargs = args.size();

                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto out_shape = out[0].get_shape();

                if (auto op_annotations = concat->get_op_annotations())
                {
                    auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                    if (in_place_oi_pairs.size() > 0)
                    {
                        auto out_size = shape_size(out_shape) * element_size;

                        auto functor =
                            [&, arg_buffer_indices, nargs, out_size, arg_sizes, out_buffer_index](
                                CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                                auto offset = 0;
                                for (size_t i = 0; i < nargs; i++)
                                {
                                    // if the argument pointer does not fall within the concat
                                    // output buffer (caused by propagate_in_place_output or
                                    // propagate_in_place_input), we need to copy the data;
                                    // otherwise, we can skip the copy.
                                    if (ctx->buffer_data[arg_buffer_indices[i]] <
                                            ctx->buffer_data[out_buffer_index] ||
                                        ctx->buffer_data[arg_buffer_indices[i]] >=
                                            reinterpret_cast<char*>(
                                                ctx->buffer_data[out_buffer_index]) +
                                                out_size)
                                    {
                                        memcpy(reinterpret_cast<char*>(
                                                   ctx->buffer_data[out_buffer_index]) +
                                                   offset,
                                               ctx->buffer_data[arg_buffer_indices[i]],
                                               arg_sizes[i]);
                                    }
                                    offset += arg_sizes[i];
                                }

                            };

                        functors.emplace_back(functor);
                        return;
                    }
                }

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto concat_pd =
                        mkldnn_emitter->get_concat_desc<ngraph::op::Concat>(node, nargs);
                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0; i < nargs; i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }
                    // Concat needs number of inputs plus 2 primitives; those two are for result and
                    // concat.
                    auto concat_index = mkldnn_emitter->reserve_primitive_space(nargs + 2);
                    auto& deps = mkldnn_emitter->get_primitive_deps(concat_index);

                    auto functor = [&,
                                    concat_pd,
                                    inputs_data_desc,
                                    arg_buffer_indices,
                                    nargs,
                                    concat_index,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_concat(ctx->mkldnn_primitives,
                                                         concat_pd,
                                                         inputs_data_desc,
                                                         deps,
                                                         concat_index);
                        }
                        for (size_t i = 0; i < nargs; i++)
                        {
                            cpu::mkldnn_utils::set_memory_ptr(
                                ctx, deps[i], ctx->buffer_data[arg_buffer_indices[i]]);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[nargs], ctx->buffer_data[out_buffer_index]);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, concat_index);
                    };

                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::concat<float, 1>)> kernel;

                    SELECT_KERNEL_BY_RANK(kernel,
                                          out[0].get_element_type(),
                                          out[0].get_shape().size(),
                                          runtime::cpu::kernel::concat);

                    auto functor = [&,
                                    kernel,
                                    arg_buffer_indices,
                                    arg_shapes,
                                    out_shape,
                                    axis,
                                    out_buffer_index](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {
                        std::vector<void*> arg_tensors;
                        for (auto& arg_buffer_index : arg_buffer_indices)
                        {
                            arg_tensors.push_back(ctx->buffer_data[arg_buffer_index]);
                        }
                        kernel(arg_tensors,
                               arg_shapes,
                               ctx->buffer_data[out_buffer_index],
                               out_shape,
                               axis);
                    };
                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(Concat);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_concat_cpp() {}
#endif
        }
    }
}
