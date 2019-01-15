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

#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/relu.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::LeakyRelu)
            {
                auto& functors = external_function->get_functors();

                auto& input_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                size_t count = out[0].get_size();

                auto alpha = static_cast<const op::LeakyRelu*>(node)->get_alpha();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto leaky_relu_index =
                        mkldnn_emitter->build_leaky_relu(input_desc, result_desc, alpha);
                    auto& deps = mkldnn_emitter->get_primitive_deps(leaky_relu_index);
                    auto functor = [&, leaky_relu_index](CPURuntimeContext* ctx,
                                                         CPUExecutionContext* ectx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], input_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, leaky_relu_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::leaky_relu<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::leaky_relu);

                    auto functor = [&, kernel, alpha, count](CPURuntimeContext* ctx,
                                                             CPUExecutionContext* ectx) {
                        kernel(input_tensor, out_tensor, alpha, count, ectx->arena);
                    };
                    functors.emplace_back(functor);
                }
            }
            REGISTER_OP_BUILDER(LeakyRelu);
        }
    }
}
