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

#include "ngraph/runtime/cpu/op/rnn.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::Rnn)
            {
                if (!runtime::cpu::mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error(
                        "Rnn is supported only through MKLDNN and doesnt have reference "
                        "INTERPRETER implementation");
                }
                const ngraph::op::Rnn* rnn_node = static_cast<const ngraph::op::Rnn*>(node);

                const int src_sequence_length_max = rnn_node->get_src_sequence_length();
                const int direction = rnn_node->get_direction();
                const int num_fused_layers = rnn_node->get_num_fused_layers();
                const int rnn_cell_n_gates = rnn_node->get_gates_per_cell();
                const int rnn_cell_n_states = rnn_node->get_num_cell_states();
                const int feature_size = rnn_node->get_src_iter_feature_size();
                const int batch = rnn_node->get_batch_size();

                if (out[0].get_shape().size() == 2 && (out[0].get_shape()[1] != feature_size))
                {
                    throw ngraph_error(
                        "input slc{ht} feature size is not equal to output dlc{ht} feature size ");
                }

                if (out[1].get_shape().size() == 2 && (out[1].get_shape()[1] != feature_size) &&
                    rnn_node->get_num_timesteps() != 1)
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

                mkldnn::memory::dims src_layer_tz = {
                    src_sequence_length_max, batch, rnn_node->get_src_layer_feature_size()};
                mkldnn::memory::dims src_iter_tz = {
                    num_fused_layers, direction, rnn_cell_n_states, batch, feature_size};
                mkldnn::memory::dims weights_layer_tz = {num_fused_layers,
                                                         direction,
                                                         rnn_node->get_src_layer_feature_size(),
                                                         rnn_cell_n_gates,
                                                         feature_size};
                mkldnn::memory::dims weights_iter_tz = {
                    num_fused_layers, direction, feature_size, rnn_cell_n_gates, feature_size};
                mkldnn::memory::dims bias_tz = {
                    num_fused_layers, direction, rnn_cell_n_gates, feature_size};
                mkldnn::memory::dims dst_layer_tz = {src_sequence_length_max, batch, feature_size};
                mkldnn::memory::dims dst_iter_tz = {
                    num_fused_layers, direction, rnn_cell_n_states, batch, feature_size};

                // We create the memory descriptors used by the user
                auto src_layer_md = mkldnn::memory::desc(
                    {src_layer_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::tnc);

                auto src_iter_md = mkldnn::memory::desc(
                    {src_iter_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::ldsnc);

                auto wei_layer_md = mkldnn::memory::desc({weights_layer_tz},
                                                         mkldnn::memory::data_type::f32,
                                                         mkldnn::memory::format::ldigo);

                auto wei_iter_md = mkldnn::memory::desc({weights_iter_tz},
                                                        mkldnn::memory::data_type::f32,
                                                        mkldnn::memory::format::ldigo);

                auto bias_md = mkldnn::memory::desc(
                    {bias_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::ldgo);

                auto dst_layer_md = mkldnn::memory::desc(
                    {dst_layer_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::tnc);

                auto dst_iter_md = mkldnn::memory::desc(
                    {dst_iter_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::ldsnc);

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto rnn_index = mkldnn_emitter->build_rnn_forward(src_layer_md,
                                                                   src_iter_md,
                                                                   wei_layer_md,
                                                                   wei_iter_md,
                                                                   bias_md,
                                                                   dst_layer_md,
                                                                   dst_iter_md);
                auto& deps = mkldnn_emitter->get_primitive_deps(rnn_index);

                auto functor = [&, rnn_index](CPURuntimeContext* ctx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], src_layer_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], src_iter_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], weights_layer_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], weights_iter_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[4], bias_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[5], dst_layer_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[6], dst_iter_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(
                        ctx, deps[7], ctx->mkldnn_workspaces[deps[8]]);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, rnn_index);
                };
                functors.emplace_back(functor);
            }
            REGISTER_OP_BUILDER(Rnn);
        }
    }
}
