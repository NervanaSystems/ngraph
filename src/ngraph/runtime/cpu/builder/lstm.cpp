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
                        "Lstm op doesnt have the required number of inputs to emit MKLDNN kernel");
                }
                auto src_sequence_length_max =
                    static_cast<unsigned long>(lstm_node->get_src_sequence_length());
                auto direction = static_cast<unsigned long>(lstm_node->get_direction());
                auto num_fused_layers =
                    static_cast<unsigned long>(lstm_node->get_num_fused_layers());
                auto feature_size =
                    static_cast<unsigned long>(lstm_node->get_src_iter_feature_size());
                auto batch = static_cast<unsigned long>(lstm_node->get_batch_size());
                auto lstm_cell_n_gates =
                    static_cast<unsigned long>(lstm_node->get_gates_per_cell());
                auto lstm_cell_n_states =
                    static_cast<unsigned long>(lstm_node->get_num_cell_states());

                if (out[0].get_shape().size() == 2 && (out[0].get_shape()[1] != feature_size))
                {
                    throw ngraph_error(
                        "input slc{ht} feature size is not equal to output dlc{ht} feature size ");
                }

                if (out[1].get_shape().size() == 2 && (out[1].get_shape()[1] != feature_size) &&
                    lstm_node->get_num_timesteps() != 1)
                {
                    throw ngraph_error(
                        "input sic{ht_1|ct_1} feature size is not equal to output dlc{ht_1|ct_1} "
                        "feature size ");
                }

                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& src_layer_tensor = tensor_data[args[0].get_name()];
                auto& src_iter_tensor = tensor_data[args[1].get_name()];
                auto& weights_layer_tensor = tensor_data[args[2].get_name()];
                auto& weights_iter_tensor = tensor_data[args[3].get_name()];
                auto& bias_tensor = tensor_data[args[4].get_name()];
                auto& dst_layer_tensor = tensor_data[out[0].get_name()];
                auto& dst_iter_tensor = tensor_data[out[1].get_name()];

                Shape src_layer_tz{
                    src_sequence_length_max,
                    batch,
                    static_cast<unsigned long>(lstm_node->get_src_layer_feature_size())};
                Shape src_iter_tz{
                    num_fused_layers, direction, lstm_cell_n_states, batch, feature_size};
                Shape wei_layer_tz{
                    num_fused_layers,
                    direction,
                    static_cast<unsigned long>(lstm_node->get_src_layer_feature_size()),
                    lstm_cell_n_gates,
                    feature_size};
                Shape wei_iter_tz{
                    num_fused_layers, direction, feature_size, lstm_cell_n_gates, feature_size};
                Shape bias_tz{num_fused_layers, direction, lstm_cell_n_gates, feature_size};
                Shape dst_layer_tz{src_sequence_length_max, batch, feature_size};
                Shape dst_iter_tz{
                    num_fused_layers, direction, lstm_cell_n_states, batch, feature_size};

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                // We create the memory descriptors used by the user
                auto src_layer_md = mkldnn_emitter->build_memory_descriptor(
                    src_layer_tz, args[0].get_element_type(), mkldnn::memory::format::tnc);
                auto src_iter_md = mkldnn_emitter->build_memory_descriptor(
                    src_iter_tz, args[1].get_element_type(), mkldnn::memory::format::ldsnc);
                auto wei_layer_md = mkldnn_emitter->build_memory_descriptor(
                    wei_layer_tz, args[2].get_element_type(), mkldnn::memory::format::ldigo);
                auto wei_iter_md = mkldnn_emitter->build_memory_descriptor(
                    wei_iter_tz, args[3].get_element_type(), mkldnn::memory::format::ldigo);
                auto bias_md = mkldnn_emitter->build_memory_descriptor(
                    bias_tz, args[4].get_element_type(), mkldnn::memory::format::ldgo);
                auto dst_layer_md = mkldnn_emitter->build_memory_descriptor(
                    dst_layer_tz, out[0].get_element_type(), mkldnn::memory::format::tnc);
                auto dst_iter_md = mkldnn_emitter->build_memory_descriptor(
                    dst_iter_tz, out[1].get_element_type(), mkldnn::memory::format::ldsnc);

                auto lstm_index = mkldnn_emitter->build_rnn_forward(src_layer_md,
                                                                    src_iter_md,
                                                                    wei_layer_md,
                                                                    wei_iter_md,
                                                                    bias_md,
                                                                    dst_layer_md,
                                                                    dst_iter_md);
                auto& deps = mkldnn_emitter->get_primitive_deps(lstm_index);

                auto functor = [&, lstm_index](CPURuntimeContext* ctx) {
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
