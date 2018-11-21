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

#include "ngraph/runtime/cpu/op/lstm.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Lstm)
            {
                if (!runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error(
                        "Lstm is supported only through MKLDNN and doesnt have reference "
                        "INTERPRETER implementation");
                }

                const ngraph::op::Lstm* lstm_node = static_cast<const ngraph::op::Lstm*>(node);
                if (args.size() != 5 || !lstm_node->get_fused_inputs())
                {
                    throw ngraph_error(
                        "Lstm op doesnt have the required number of inputs to create MKLDNN "
                        "kernel");
                }
                auto& functors = external_function->get_functors();

                auto& src_layer_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& src_iter_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& weights_layer_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& weights_iter_tensor = external_function->get_tensor_data(args[3].get_name());
                auto& bias_tensor = external_function->get_tensor_data(args[4].get_name());
                auto& dst_layer_tensor = external_function->get_tensor_data(out[0].get_name());
                auto& dst_iter_tensor = external_function->get_tensor_data(out[1].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto lstm_index = mkldnn_emitter->build_rnn<ngraph::op::Lstm>(node, args, out);
                auto& deps = mkldnn_emitter->get_primitive_deps(lstm_index);

                auto functor = [&, lstm_index](CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], src_layer_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], src_iter_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], weights_layer_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], weights_iter_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[4], bias_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[5], dst_layer_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[6], dst_iter_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[7], ctx->mkldnn_workspaces[deps[8]]);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, lstm_index);
                };
                functors.emplace_back(functor);
            }
            REGISTER_OP_BUILDER(Lstm);
        }
    }
}
