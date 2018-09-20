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

#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/softmax.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Softmax)
            {
                auto softmax = static_cast<const ngraph::op::Softmax*>(node);

                auto& functors = external_function->get_functors();

                auto arg_shape = args[0].get_shape();

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto axes = softmax->get_axes();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    if (axes.size() != 1)
                    {
                        throw ngraph_error("MKLDNN supports softmax only across single axis");
                    }

                    int softmax_axis = static_cast<int>(*(axes.begin()));
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t softmax_index = mkldnn_emitter->build_softmax_forward(
                        input_desc, result_desc, softmax_axis);

                    auto& deps = mkldnn_emitter->get_primitive_deps(softmax_index);

                    auto functor = [&, softmax_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, softmax_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    if (axes.size() == arg_shape.size())
                    {
                        std::function<decltype(runtime::cpu::kernel::softmax_all<float, 1>)> kernel;

                        PARTIAL_SELECT_KERNEL_BY_RANK(kernel,
                                                      args[0].get_element_type(),
                                                      args[0].get_shape().size(),
                                                      runtime::cpu::kernel::softmax_all);

                        auto functor = [&, kernel, arg_shape](CPURuntimeContext* ctx) {
                            kernel(arg_tensor, out_tensor, arg_shape);
                        };
                        functors.emplace_back(functor);
                    }
                    else if (axes.size() == 1)
                    {
                        if (*axes.begin() == (arg_shape.size() - 1))
                        {
                            std::function<decltype(
                                runtime::cpu::kernel::softmax_innermost_1rd<float, 1>)>
                                kernel;

                            PARTIAL_SELECT_KERNEL_BY_RANK(
                                kernel,
                                args[0].get_element_type(),
                                args[0].get_shape().size(),
                                runtime::cpu::kernel::softmax_innermost_1rd);

                            auto functor = [&, kernel, arg_shape](CPURuntimeContext* ctx) {
                                kernel(arg_tensor, out_tensor, arg_shape);
                            };
                            functors.emplace_back(functor);
                        }
                        else
                        {
                            std::function<decltype(runtime::cpu::kernel::softmax_1rd<float, 1>)>
                                kernel;

                            PARTIAL_SELECT_KERNEL_BY_RANK(kernel,
                                                          args[0].get_element_type(),
                                                          args[0].get_shape().size(),
                                                          runtime::cpu::kernel::softmax_1rd);

                            auto functor = [&, kernel, arg_shape, axes](CPURuntimeContext* ctx) {
                                kernel(arg_tensor, out_tensor, arg_shape, axes);
                            };
                            functors.emplace_back(functor);
                        }
                    }
                    else if (arg_shape.size() == 3 && axes.size() == 2)
                    {
                        std::function<decltype(runtime::cpu::kernel::softmax_3d_2rd<float>)> kernel;

                        SELECT_KERNEL(kernel,
                                      args[0].get_element_type(),
                                      runtime::cpu::kernel::softmax_3d_2rd);

                        auto functor = [&, kernel, arg_shape, axes](CPURuntimeContext* ctx) {
                            kernel(arg_tensor, out_tensor, arg_shape, axes);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        throw ngraph_error("Unsupported Softmax");
                    }
                }
            }

            REGISTER_OP_BUILDER(Softmax);
        }
    }
}
