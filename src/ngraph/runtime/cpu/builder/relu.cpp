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

#include "ngraph/runtime/cpu/kernel/relu.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Relu)
            {
                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& functors = external_function->get_functors();

                    auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                    auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t relu_index = mkldnn_emitter->build_relu_forward(input_desc, result_desc);

                    auto& deps = mkldnn_emitter->get_primitive_deps(relu_index);

                    auto functor = [&, relu_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, relu_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    BUILD_UNARY_ELEMWISE_FUNCTOR(runtime::cpu::kernel::relu);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::ReluBackprop)
            {
                auto& functors = external_function->get_functors();

                auto& arg_fwd_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& delta_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());
                size_t count = out[0].get_size();

                if (runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t relu_index =
                        mkldnn_emitter->build_relu_backward(input_desc, delta_desc, result_desc);

                    auto& deps = mkldnn_emitter->get_primitive_deps(relu_index);
                    auto functor = [&, relu_index](CPURuntimeContext* ctx) {
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_fwd_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], delta_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, relu_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    std::function<decltype(runtime::cpu::kernel::relu_backprop<float>)> kernel;

                    SELECT_KERNEL(
                        kernel, out[0].get_element_type(), runtime::cpu::kernel::relu_backprop);

                    auto functor = [&, kernel, count](CPURuntimeContext* ctx) {
                        kernel(arg_fwd_tensor, delta_tensor, out_tensor, count);
                    };
                    functors.emplace_back(functor);
                }
            }

            REGISTER_OP_BUILDER(Relu);
            REGISTER_OP_BUILDER(ReluBackprop);
        }
    }
}
