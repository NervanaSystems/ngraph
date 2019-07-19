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

#include "ngraph/op/experimental/quantized_concat.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::QuantizedConcat)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();
                    vector<size_t> arg_buffer_indices;
                    for (auto& arg : args)
                    {
                        if (shape_size(arg.get_shape()))
                        {
                            arg_buffer_indices.emplace_back(
                                external_function->get_buffer_index(arg.get_name()));
                        }
                    }
                    auto nargs = args.size();

                    auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto concat_pd =
                        mkldnn_emitter->get_concat_desc<ngraph::op::QuantizedConcat>(node, nargs);
                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0; i < args.size(); i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    // Concat needs number of inputs plus 2 primitives; those two are for result and concat.
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
                    throw ngraph_error("unsupported parameters for QuantizedConcat via DEX");
                }
            }
            REGISTER_OP_BUILDER(QuantizedConcat);
        }
    }
}
