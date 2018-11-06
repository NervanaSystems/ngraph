//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

                vector<reference_wrapper<void*>> arg_tensors;
                vector<Shape> arg_shapes;
                for (auto& arg : args)
                {
                    if (shape_size(arg.get_shape()))
                    {
                        arg_tensors.emplace_back(
                            external_function->get_tensor_data(arg.get_name()));
                        arg_shapes.emplace_back(arg.get_shape());
                    }
                }
                auto nargs = args.size();

                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                auto out_shape = out[0].get_shape();

                auto element_size = concat->get_input_element_type(0).size();
                if (auto op_annotations = concat->get_op_annotations())
                {
                    auto in_place_oi_pairs = op_annotations->get_in_place_oi_pairs();
                    if (in_place_oi_pairs.size() > 0)
                    {
                        auto functor = [&, arg_tensors, nargs, out_shape, arg_shapes, element_size](
                            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                            auto out_size = shape_size(out_shape) * element_size;
                            auto offset = 0;
                            for (size_t i = 0; i < nargs; i++)
                            {
                                auto arg_size = shape_size(arg_shapes[i]) * element_size;
                                // if the argument pointer does not fall within the concat output buffer
                                // (caused by propagate_in_place_output or propagate_in_place_input), we need to copy the data;
                                // otherwise, we can skip the copy.
                                if (arg_tensors[i] < out_tensor ||
                                    arg_tensors[i] >=
                                        reinterpret_cast<char*>(out_tensor) + out_size)
                                {
                                    memcpy(reinterpret_cast<char*>(out_tensor) + offset,
                                           arg_tensors[i],
                                           arg_size);
                                }
                                offset += arg_size;
                            }

                        };

                        functors.emplace_back(functor);
                        return;
                    }
                }

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0; i < args.size(); i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim =
                        (dynamic_cast<const ngraph::op::Concat*>(node))->get_concatenation_axis();
                    auto concat_index =
                        mkldnn_emitter->build_concat(inputs_data_desc, result_desc, concat_dim);
                    auto& deps = mkldnn_emitter->get_primitive_deps(concat_index);

                    auto functor = [&, arg_tensors, nargs, concat_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        for (size_t i = 0; i < nargs; i++)
                        {
                            cpu::mkldnn_utils::set_memory_ptr(ctx, deps[i], arg_tensors[i]);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[nargs], out_tensor);
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

                    auto functor = [&, kernel, arg_tensors, arg_shapes, out_shape, axis](
                        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        kernel(arg_tensors, arg_shapes, out_tensor, out_shape, axis);
                    };
                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(Concat);
        }
    }
}
