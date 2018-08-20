/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/lrn.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/reference/lrn.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::LRN)
            {
                auto& functors = external_function->get_functors();

                const ngraph::op::LRN* lrn = static_cast<const ngraph::op::LRN*>(node);
                function<void(CPURuntimeContext*)> functor;

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto lrn_index =
                        mkldnn_emitter->build_lrn_forward(input_data_desc,
                                                          result_desc,
                                                          static_cast<float>(lrn->get_alpha()),
                                                          static_cast<float>(lrn->get_beta()),
                                                          static_cast<float>(lrn->get_bias()),
                                                          static_cast<int>(lrn->get_nsize()));

                    auto& deps = mkldnn_emitter->get_primitive_deps(lrn_index);
                    functor = [&, lrn_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, lrn_index);
                    };
                }
                else
                {
                    double alpha = lrn->get_alpha();
                    double beta = lrn->get_beta();
                    double bias = lrn->get_bias();
                    double nsize = lrn->get_nsize();
                    Shape arg_shape = args[0].get_shape();

                    auto element_type = lrn->get_element_type();
                    if (element_type == element::f32)
                    {
                        functor = [&, alpha, beta, bias, arg_shape, nsize](CPURuntimeContext* ctx) {
                            ngraph::runtime::reference::lrn<float>(static_cast<float*>(arg_tensor),
                                                                   static_cast<float*>(out_tensor),
                                                                   arg_shape,
                                                                   alpha,
                                                                   beta,
                                                                   bias,
                                                                   nsize);
                        };
                    }
                    else if (element_type == element::f64)
                    {
                        functor = [&, alpha, beta, bias, arg_shape, nsize](CPURuntimeContext* ctx) {
                            ngraph::runtime::reference::lrn<double>(
                                static_cast<double*>(arg_tensor),
                                static_cast<double*>(out_tensor),
                                arg_shape,
                                alpha,
                                beta,
                                bias,
                                nsize);
                        };
                    }
                    else
                    {
                        throw ngraph_error("Unsupported type in CPU Builder for LRN");
                    }
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(LRN);
        }
    }
}
