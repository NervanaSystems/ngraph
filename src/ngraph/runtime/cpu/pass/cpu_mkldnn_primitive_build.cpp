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

#include <string>

#include "cpu_mkldnn_primitive_build.hpp"

#include "ngraph/code_writer.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_concat.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/runtime/cpu/mkldnn_emitter.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"

using namespace ngraph;
using namespace ngraph::op;
using namespace ngraph::runtime::cpu;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                // serialize memory descriptors
                static void serialize_memory_descs(std::ofstream& desc_file,
                                                   std::vector<mkldnn::memory::desc>& descs,
                                                   size_t primitive_index)
                {
                    for (auto i = 0; i < descs.size(); i++)
                    {
                        desc_file << primitive_index;
                        desc_file.write(reinterpret_cast<char*>(&descs[i]),
                                        sizeof(mkldnn::memory::desc));
                        primitive_index++;
                    }
                }

                // The following functions build the MKLDNN primitive for each type of nGraph Node.

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Add)
                {
                    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    // Add needs 4 primitives: input0, input1, result, and sum.
                    index = mkldnn_emitter.reserve_primitive_space_cg(4);
                    deps = mkldnn_emitter.get_primitive_deps_cg(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<mkldnn::memory::desc> descs = {
                        input0_data_desc, input1_data_desc, result_desc};
                    auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                    mkldnn_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "std::vector<float> scale_vector(2, 1);\n";
                    writer << "std::vector<mkldnn::memory::primitive_desc> inputs_pd;\n";
                    writer << "inputs_pd.push_back(mkldnn::memory::primitive_desc("
                              "*cg_ctx->mkldnn_descriptors["
                           << desc_index << "], "
                                            "cg_ctx->global_cpu_engine));\n";
                    writer << "inputs_pd.push_back(mkldnn::memory::primitive_desc("
                              "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 1 << "], "
                                                "cg_ctx->global_cpu_engine));\n";

                    // elementwise sum primitive descriptor
                    writer << "mkldnn::sum::primitive_desc sum_pd = "
                              "mkldnn::sum::primitive_desc(*cg_ctx->mkldnn_descriptors["
                           << desc_index + 2 << "], "
                                                "scale_vector, inputs_pd);\n";

                    writer << "\n// build sum primitive\n";
                    writer << "std::vector<mkldnn::memory::primitive::at> inputs_primitive;\n";
                    //emit_memory_primitive_build(writer, desc_names, deps);
                    writer << "inputs_primitive.push_back(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[0]) << "]);\n";
                    writer << "inputs_primitive.push_back(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[1]) << "]);\n";

                    // sum primitive
                    writer << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                           << "] = new mkldnn::sum(sum_pd, inputs_primitive, "
                              "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[2]) << "]);\n";

                    construct_string = writer.get_code();
                }

                template <typename OP>
                void construct_primitive_build_string_rnn(
                    ngraph::runtime::cpu::MKLDNNEmitter& mkldnn_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    std::ofstream& desc_file)
                {
                    const auto& out = node->get_outputs();
                    const auto& args = node->get_inputs();
                    auto rnn_node = static_cast<const OP*>(node);
                    auto src_sequence_length_max =
                        static_cast<unsigned long>(rnn_node->get_src_sequence_length());
                    auto direction = static_cast<unsigned long>(rnn_node->get_direction());
                    auto num_fused_layers =
                        static_cast<unsigned long>(rnn_node->get_num_fused_layers());
                    auto feature_size =
                        static_cast<unsigned long>(rnn_node->get_src_iter_feature_size());
                    auto batch = static_cast<unsigned long>(rnn_node->get_batch_size());
                    auto rnn_cell_n_gates =
                        static_cast<unsigned long>(rnn_node->get_gates_per_cell());
                    auto rnn_cell_n_states =
                        static_cast<unsigned long>(rnn_node->get_num_cell_states());

                    auto get_mkldnn_rnn_cell_type = [&]() {
                        switch (rnn_node->get_rnn_type())
                        {
                        case rnn_utils::rnntype::vanilla_rnn:
                            return std::string("mkldnn::algorithm::vanilla_rnn");
                        case rnn_utils::rnntype::vanilla_gru:
                            return std::string("mkldnn::algorithm::vanilla_gru");
                        case rnn_utils::rnntype::vanilla_lstm:
                            return std::string("mkldnn::algorithm::vanilla_lstm");
                        default: throw ngraph_error("unsupported mkldnn rnn algorithm");
                        }
                    };

                    auto get_mkldnn_rnn_direction = [&]() {
                        switch (direction)
                        {
                        case 1:
                            return std::string("mkldnn::rnn_direction::unidirectional_left2right");
                        case 2: return std::string("mkldnn::rnn_direction::bidirectional_concat");
                        default: throw ngraph_error("unsupported mkldnn rnn direction");
                        }
                    };

                    if (out[0].get_shape().size() == 2 &&
                        (out[0].get_shape()[1] != direction * feature_size))
                    {
                        throw ngraph_error(
                            "input slc{ht} feature size is not equal to output dlc{ht} feature "
                            "size ");
                    }

                    if (out[1].get_shape().size() == 2 && (out[1].get_shape()[1] != feature_size) &&
                        rnn_node->get_num_timesteps() != 1)
                    {
                        throw ngraph_error(
                            "input sic{ht_1|ct_1} feature size is not equal to output "
                            "dlc{ht_1|ct_1} "
                            "feature size ");
                    }

                    Shape src_layer_tz{
                        src_sequence_length_max,
                        batch,
                        static_cast<unsigned long>(rnn_node->get_src_layer_feature_size())};
                    Shape src_iter_tz{
                        num_fused_layers, direction, rnn_cell_n_states, batch, feature_size};
                    Shape wei_layer_tz{
                        num_fused_layers,
                        direction,
                        static_cast<unsigned long>(rnn_node->get_src_layer_feature_size()),
                        rnn_cell_n_gates,
                        feature_size};
                    Shape wei_iter_tz{
                        num_fused_layers, direction, feature_size, rnn_cell_n_gates, feature_size};
                    Shape bias_tz{num_fused_layers, direction, rnn_cell_n_gates, feature_size};
                    Shape dst_layer_tz{src_sequence_length_max, batch, direction * feature_size};
                    Shape dst_iter_tz{
                        num_fused_layers, direction, rnn_cell_n_states, batch, feature_size};

                    // We create the memory descriptors used by the user
                    auto src_layer_md = mkldnn_emitter.build_memory_descriptor(
                        src_layer_tz, args[0].get_element_type(), mkldnn::memory::format::tnc);
                    auto src_iter_md = mkldnn_emitter.build_memory_descriptor(
                        src_iter_tz, args[1].get_element_type(), mkldnn::memory::format::ldsnc);
                    auto wei_layer_md = mkldnn_emitter.build_memory_descriptor(
                        wei_layer_tz, args[2].get_element_type(), mkldnn::memory::format::ldigo);
                    auto wei_iter_md = mkldnn_emitter.build_memory_descriptor(
                        wei_iter_tz, args[3].get_element_type(), mkldnn::memory::format::ldigo);
                    auto bias_md = mkldnn_emitter.build_memory_descriptor(
                        bias_tz, args[4].get_element_type(), mkldnn::memory::format::ldgo);
                    auto dst_layer_md = mkldnn_emitter.build_memory_descriptor(
                        dst_layer_tz, out[0].get_element_type(), mkldnn::memory::format::tnc);
                    auto dst_iter_md = mkldnn_emitter.build_memory_descriptor(
                        dst_iter_tz, out[1].get_element_type(), mkldnn::memory::format::ldsnc);

                    // Lstm/Rnn needs 9 primitives: src_layer, src_iter, weights_layer, weights_iter, bias,
                    // dst_layer, dst_iter, workspace, and rnn_forward.
                    // It needs a new workspace.
                    index = mkldnn_emitter.reserve_primitive_space_cg(9, true /* new workspace */);
                    deps = mkldnn_emitter.get_primitive_deps_cg(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<mkldnn::memory::desc> descs = {src_layer_md,
                                                               src_iter_md,
                                                               wei_layer_md,
                                                               wei_iter_md,
                                                               bias_md,
                                                               dst_layer_md,
                                                               dst_iter_md};
                    auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                    mkldnn_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "mkldnn::rnn_cell::desc rnn_cell_desc(" << get_mkldnn_rnn_cell_type()
                           << ");\n";
                    writer << "\n// build lstm/rnn primitive descriptor\n";
                    writer << "auto rnn_desc = "
                              "mkldnn::rnn_forward::desc(mkldnn::prop_kind::forward_training, "
                              "rnn_cell_desc, "
                           << get_mkldnn_rnn_direction() << ", "
                                                            "*cg_ctx->mkldnn_descriptors["
                           << desc_index << "], "
                                            "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 1 << "], "
                                                "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 2 << "], "
                                                "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 3 << "], "
                                                "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 4 << "], "
                                                "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 5 << "], "
                                                "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 6 << "]);\n";

                    writer << "auto rnn_prim_desc = mkldnn::rnn_forward::primitive_desc(rnn_desc, "
                              "cg_ctx->global_cpu_engine);\n";
                    writer << "cg_ctx->mkldnn_primitives[" << std::to_string(deps[7])
                           << "] = new "
                              "mkldnn::memory({rnn_prim_desc.workspace_primitive_desc().desc(), "
                              "cg_ctx->global_cpu_engine}, nullptr);\n";
                    writer << "auto workspace = "
                              "(char*)malloc(rnn_prim_desc.workspace_primitive_desc().get_size());"
                              "\n";
                    writer << "if (!workspace)\n";
                    writer.block_begin();
                    writer << "throw std::bad_alloc();\n";
                    writer.block_end();
                    writer << "cg_ctx->mkldnn_workspaces.push_back(workspace);\n";

                    deps[8] = mkldnn_emitter.reserve_workspace();

                    writer << "\n// build lstm/rnn primitive\n";
                    // lstm/rnn primitive
                    writer << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                           << "] = new mkldnn::rnn_forward(rnn_prim_desc, "
                              "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[0])
                           << "]), "
                              "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[1])
                           << "]), "
                              "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[2])
                           << "]), "
                              "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[3])
                           << "]), "
                              "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[4])
                           << "]), "
                              "static_cast<mkldnn::memory>(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[5])
                           << "]), "
                              "static_cast<mkldnn::memory>(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[6])
                           << "]), "
                              "static_cast<mkldnn::memory>(*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[7]) << "]));\n";

                    construct_string = writer.get_code();
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Lstm)
                {
                    construct_primitive_build_string_rnn<Lstm>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Rnn)
                {
                    construct_primitive_build_string_rnn<Rnn>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <typename OP>
                void construct_primitive_build_string_batchnorm(
                    ngraph::runtime::cpu::MKLDNNEmitter& mkldnn_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    std::ofstream& desc_file,
                    const bool append_relu,
                    const bool training)
                {
                    const auto& args = node->get_inputs();

                    // batchnorm forward needs 6 primitives: input, weights, result, mean,
                    // variance, and batch_normalization_forward.
                    index = mkldnn_emitter.reserve_primitive_space_cg(6);
                    deps = mkldnn_emitter.get_primitive_deps_cg(index);

                    CodeWriter writer;

                    if (append_relu)
                    {
                        writer << "mkldnn::post_ops pops;\n";
                        writer << "const float ops_scale = 1.f;\n";
                        writer << "const float ops_alpha = -0.f; // relu negative slope\n";
                        writer << "const float ops_beta = 0.f;\n";

                        writer << "pops.append_eltwise("
                                  "ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, "
                                  "ops_beta);\n";
                    }
                    else
                    {
                        writer << "mkldnn::post_ops pops = mkldnn::post_ops();\n";
                    }

                    auto weights_shape =
                        Shape{2, args[0].get_tensor().get_tensor_layout()->get_size()};
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto weights_desc = mkldnn_emitter.build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    bool use_global_stats;
                    const mkldnn::memory::desc *mean_desc, *variance_desc;
                    if (training && args.size() == 3)
                    {
                        mean_desc = &mkldnn_utils::get_output_mkldnn_md(node, 1);
                        variance_desc = &mkldnn_utils::get_output_mkldnn_md(node, 2);
                        use_global_stats = false;
                    }
                    else
                    {
                        mean_desc = &mkldnn_utils::get_input_mkldnn_md(node, 3);
                        variance_desc = &mkldnn_utils::get_input_mkldnn_md(node, 4);
                        use_global_stats = true;
                    }

                    auto batchnorm = static_cast<const OP*>(node);
                    auto eps = batchnorm->get_eps_value();

                    writer << "mkldnn::primitive_attr bn_attr;\n";
                    writer << "bn_attr.set_post_ops(pops);\n";

                    writer << "\n// build batchnorm primitive descriptor\n";
                    if (use_global_stats)
                    {
                        // Write memory descriptors to file;
                        std::vector<mkldnn::memory::desc> descs = {
                            input_desc, *mean_desc, *variance_desc, weights_desc, result_desc};
                        auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                        mkldnn_emitter.reserve_descriptor_space(descs.size());
                        serialize_memory_descs(desc_file, descs, deps[0]);

                        writer << "auto batchnorm_desc = "
                                  "mkldnn::batch_normalization_forward::desc(mkldnn::prop_kind::"
                                  "forward_training, "
                                  "*cg_ctx->mkldnn_descriptors["
                               << desc_index << "], " << eps
                               << ", "
                                  "mkldnn::batch_normalization_flag::use_scale_shift | "
                                  "mkldnn::batch_normalization_flag::use_global_stats);\n";

                        writer << "auto batchnorm_prim_desc = "
                                  "mkldnn::batch_normalization_forward::primitive_desc(batchnorm_"
                                  "desc, "
                                  "bn_attr, cg_ctx->global_cpu_engine);\n";

                        writer << "\n// build batchnorm primitive\n";
                        // batchnorm primitive
                        writer
                            << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                            << "] = new mkldnn::batch_normalization_forward(batchnorm_prim_desc, "
                               "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[0])
                            << "]), "
                               "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[1])
                            << "]), "
                               "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[2])
                            << "]), "
                               "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[3])
                            << "]), "
                               "static_cast<mkldnn::memory>(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[4]) << "]));\n";
                    }
                    else
                    {
                        // Write memory descriptors to file;
                        std::vector<mkldnn::memory::desc> descs = {
                            input_desc, weights_desc, result_desc, *mean_desc, *variance_desc};
                        auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                        mkldnn_emitter.reserve_descriptor_space(descs.size());
                        serialize_memory_descs(desc_file, descs, deps[0]);

                        writer << "auto batchnorm_desc = "
                                  "mkldnn::batch_normalization_forward::desc(mkldnn::prop_kind::"
                                  "forward_training, "
                                  "*cg_ctx->mkldnn_descriptors["
                               << desc_index << "], " << eps
                               << ", "
                                  "mkldnn::batch_normalization_flag::use_scale_shift);\n";

                        writer << "auto batchnorm_prim_desc = "
                                  "mkldnn::batch_normalization_forward::primitive_desc(batchnorm_"
                                  "desc, "
                                  "bn_attr, cg_ctx->global_cpu_engine);\n";

                        writer << "\n// build batchnorm primitive\n";
                        // batchnorm primitive
                        writer
                            << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                            << "] = new mkldnn::batch_normalization_forward(batchnorm_prim_desc, "
                               "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[0])
                            << "]), "
                               "mkldnn::primitive::at(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[1])
                            << "]), "
                               "static_cast<mkldnn::memory>(*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[2]) << "]), "
                                                          "*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[3]) << "], "
                                                          "*cg_ctx->mkldnn_primitives["
                            << std::to_string(deps[4]) << "]);\n";
                    }
                    construct_string = writer.get_code();
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormTraining)
                {
                    construct_primitive_build_string_batchnorm<BatchNormTraining>(
                        mkldnn_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        desc_file,
                        false /*Append relu*/,
                        true /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormInference)
                {
                    construct_primitive_build_string_batchnorm<BatchNormInference>(
                        mkldnn_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        desc_file,
                        false /*Append relu*/,
                        false /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormTrainingRelu)
                {
                    construct_primitive_build_string_batchnorm<BatchNormTrainingRelu>(
                        mkldnn_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        desc_file,
                        true /*Append relu*/,
                        true /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormInferenceRelu)
                {
                    construct_primitive_build_string_batchnorm<BatchNormInferenceRelu>(
                        mkldnn_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        desc_file,
                        true /*Append relu*/,
                        false /*Training*/);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormTrainingBackprop)
                {
                    const auto& args = node->get_inputs();
                    const auto* batchnorm = static_cast<const BatchNormTrainingBackprop*>(node);
                    auto eps = batchnorm->get_eps_value();

                    auto weights_shape =
                        Shape{2, args[0].get_tensor().get_tensor_layout()->get_size()};
                    auto weights_desc = mkldnn_emitter.build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto mean_desc = mkldnn_utils::get_input_mkldnn_md(node, 3);
                    auto variance_desc = mkldnn_utils::get_input_mkldnn_md(node, 4);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 5);
                    auto dinput_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto dweights_desc = mkldnn_emitter.build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);

                    // batchnorm backward needs 8 primitives: weights, input, mean, variance,
                    // dinput, dweights, and batch_normalization_backward.
                    index = mkldnn_emitter.reserve_primitive_space_cg(8);
                    deps = mkldnn_emitter.get_primitive_deps_cg(index);

                    CodeWriter writer;

                    // Write memory descriptors to file;
                    std::vector<mkldnn::memory::desc> descs = {weights_desc,
                                                               input_desc,
                                                               mean_desc,
                                                               variance_desc,
                                                               delta_desc,
                                                               dinput_desc,
                                                               dweights_desc};
                    auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                    mkldnn_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "\n// build batchnorm primitives\n";
                    writer << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                           << "] = new mkldnn::batch_normalization_backward("
                           << "{{mkldnn::prop_kind::backward, "
                              "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 4 << "], "
                                                "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 1 << "], " << eps
                           << ", "
                              "mkldnn::batch_normalization_flag::use_scale_shift}, "
                              "cg_ctx->global_cpu_engine, "
                              "{{mkldnn::prop_kind::forward_training, "

                              "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 1 << "], " << eps
                           << ", "
                              "mkldnn::batch_normalization_flag::use_scale_shift}, "
                              "cg_ctx->global_cpu_engine}}, "
                              "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[1]) << "], "
                                                         "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[2]) << "], "
                                                         "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[3]) << "], "
                                                         "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[4]) << "], "
                                                         "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[0]) << "], "
                                                         "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[5]) << "], "
                                                         "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[6]) << "]);\n";

                    construct_string = writer.get_code();
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Concat)
                {
                    auto concat = static_cast<const ngraph::op::Concat*>(node);
                    size_t concat_dim = concat->get_concatenation_axis();
                    size_t nargs = node->get_inputs().size();

                    // Concat needs number of inputs plus 2 primitives; those two are for result and concat.
                    index = mkldnn_emitter.reserve_primitive_space_cg(nargs + 2);
                    deps = mkldnn_emitter.get_primitive_deps_cg(index);

                    CodeWriter writer;

                    // Write memory descriptors to file;
                    std::vector<mkldnn::memory::desc> descs;
                    for (size_t i = 0; i < nargs; i++)
                    {
                        descs.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    descs.push_back(result_desc);
                    auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                    mkldnn_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "std::vector<mkldnn::memory::primitive::at> inputs_primitive;\n";
                    writer << "std::vector<mkldnn::memory::primitive_desc> inputs_pd;\n";

                    writer << "for (size_t i = " << desc_index << "; i < " << desc_index + nargs
                           << "; i++)\n";
                    writer.block_begin();
                    writer << "inputs_pd.push_back(mkldnn::memory::primitive_desc("
                              "*cg_ctx->mkldnn_descriptors[i], "
                              "cg_ctx->global_cpu_engine));\n";
                    writer.block_end();

                    writer << "for (size_t i = " << deps[0] << "; i < " << deps[0] + nargs
                           << "; i++)\n";
                    writer.block_begin();
                    writer << "inputs_primitive.push_back(*cg_ctx->mkldnn_primitives[i]);\n";
                    writer.block_end();

                    writer << "auto concat_prim_desc = "
                              "mkldnn::concat::primitive_desc( "
                              "*cg_ctx->mkldnn_descriptors["
                           << desc_index + nargs << "], "
                           << std::to_string(static_cast<int>(concat_dim)) << ", inputs_pd);\n";

                    writer << "\n// build concat primitive\n";
                    writer << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                           << "] = new mkldnn::concat(concat_prim_desc, inputs_primitive, "
                              "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[nargs]) << "]);\n";

                    construct_string = writer.get_code();
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(LRN)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    // LRN needs 3 primitives: input, result, and lrn_forward.
                    index = mkldnn_emitter.reserve_primitive_space_cg(3);
                    deps = mkldnn_emitter.get_primitive_deps_cg(index);

                    CodeWriter writer;

                    // Write memory descriptors to file;
                    std::vector<mkldnn::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                    mkldnn_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    const auto* lrn = static_cast<const LRN*>(node);
                    auto alpha = static_cast<float>(lrn->get_alpha());
                    auto beta = static_cast<float>(lrn->get_beta());
                    auto bias = static_cast<float>(lrn->get_bias());
                    auto nsize = static_cast<int>(lrn->get_nsize());

                    writer << "auto lrn_desc = "
                              "mkldnn::lrn_forward::desc(mkldnn::prop_kind::forward_scoring, "
                              "mkldnn::algorithm::lrn_across_channels, "
                              "*cg_ctx->mkldnn_descriptors["
                           << desc_index << "], " << nsize << ", " << alpha << ", " << beta << ", "
                           << bias << ");\n";

                    writer << "auto lrn_prim_desc = "
                              "mkldnn::lrn_forward::primitive_desc(lrn_desc, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build lrn primitive\n";
                    writer << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                           << "] = new mkldnn::lrn_forward(lrn_prim_desc, "
                              "*cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[0]) << "]"
                                                         ", *cg_ctx->mkldnn_primitives["
                           << std::to_string(deps[1]) << "]);\n";

                    construct_string = writer.get_code();
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Slice)
                {
                    const auto& out = node->get_outputs();
                    const Slice* slice = static_cast<const Slice*>(node);
                    auto out_shape = out[0].get_shape();
                    auto lower_bounds = slice->get_lower_bounds();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_slice(
                        input_desc, result_desc, lower_bounds, out_shape);
                }

                template <typename OP>
                void construct_primitive_build_string_conv(
                    ngraph::runtime::cpu::MKLDNNEmitter& mkldnn_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    std::ofstream& desc_file)
                {
                    auto convolution = static_cast<const OP*>(node);
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    // MKLDNN relies on named formats for kernel selection
                    if (weights_desc.data.format == mkldnn_nchw)
                    {
                        weights_desc.data.format = mkldnn_oihw;
                    }
                    if (weights_desc.data.format == mkldnn_ncdhw)
                    {
                        weights_desc.data.format = mkldnn_oidhw;
                    }
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto strides = convolution->get_window_movement_strides();
                    auto pad_below = convolution->get_padding_below();
                    auto pad_above = convolution->get_padding_above();

                    if (mkldnn_emitter.has_bias<OP>())
                    {
                        index = mkldnn_emitter.reserve_primitive_space_cg(5);
                    }
                    else
                    {
                        index = mkldnn_emitter.reserve_primitive_space_cg(4);
                    }
                    deps = mkldnn_emitter.get_primitive_deps_cg(index);

                    CodeWriter writer;

                    writer << "// Write in memory descriptors\n";
                    std::vector<mkldnn::memory::desc> descs = {
                        data_desc, weights_desc, result_desc};

                    if (mkldnn_emitter.has_bias<OP>())
                    {
                        auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                        descs.insert(descs.begin() + 2, bias_desc);
                    }

                    auto desc_index = mkldnn_emitter.get_mkldnn_descriptors_size();
                    mkldnn_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "\n// build QConv primitive descriptor\n";
                    writer << "auto conv_desc = "
                              "mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward,\n"
                              "mkldnn::algorithm::convolution_direct,\n"
                              "*cg_ctx->mkldnn_descriptors["
                           << desc_index << "],\n"
                                            "*cg_ctx->mkldnn_descriptors["
                           << desc_index + 1 << "],\n";
                    if (mkldnn_emitter.has_bias<OP>())
                    {
                        writer << "*cg_ctx->mkldnn_descriptors[" << desc_index + 2 << "],\n";
                    }
                    writer << "*cg_ctx->mkldnn_descriptors[" << desc_index + (descs.size() - 1)
                           << "],\n"
                              "mkldnn::memory::dims{";
                    if (strides.size() > 1)
                    {
                        for (auto i = 0; i < strides.size() - 1; i++)
                        {
                            writer << std::to_string(strides[i]) << ", ";
                        }
                    }
                    writer << std::to_string(strides[strides.size() - 1]);
                    writer << "},\n"
                              "mkldnn::memory::dims{";
                    if (window_dilation_strides_adjusted.size() > 1)
                    {
                        for (auto i = 0; i < window_dilation_strides_adjusted.size() - 1; i++)
                        {
                            writer << std::to_string(window_dilation_strides_adjusted[i]) << ", ";
                        }
                    }
                    writer << std::to_string(
                        window_dilation_strides_adjusted[window_dilation_strides_adjusted.size() -
                                                         1]);
                    writer << "},\n"
                              "mkldnn::memory::dims{";
                    if (pad_below.size() > 1)
                    {
                        for (auto i = 0; i < pad_below.size() - 1; i++)
                        {
                            writer << std::to_string(pad_below[i]) << ", ";
                        }
                    }
                    writer << std::to_string(pad_below[pad_below.size() - 1]);
                    writer << "},\n"
                              "mkldnn::memory::dims{";
                    if (pad_above.size() > 1)
                    {
                        for (auto i = 0; i < pad_above.size() - 1; i++)
                        {
                            writer << std::to_string(pad_above[i]) << ", ";
                        }
                    }
                    writer << std::to_string(pad_above[pad_above.size() - 1]);
                    writer << "},\n"
                              "mkldnn::padding_kind::zero);\n";

                    writer << "mkldnn::post_ops ops;\n";
                    if (std::is_same<OP, ngraph::op::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::ConvolutionAdd>())
                    {
                        writer << "ops.append_sum(1.f);\n";
                    }

                    if (std::is_same<OP, ngraph::op::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::QuantizedConvolutionBiasSignedAdd>())
                    {
                        auto sum_scales_size = shape_size(convolution->get_input_shape(5));
                        const element::Type& et = node->get_input_element_type(5);
                        std::string type = et.c_type_string();
                        std::stringstream ss;
                        writer << "std::vector<float> dyn_post_op_scales;\n";
                        auto c = std::dynamic_pointer_cast<ngraph::op::Constant>(
                            node->get_arguments()[5]);
                        if (c)
                        {
                            auto sum_scale_val =
                                extract_scale_value<ngraph::op::QuantizedConvolutionBiasAdd>(node,
                                                                                             5);
                            writer << "dyn_post_op_scales.push_back("
                                   << std::to_string(sum_scale_val[0]) << ");\n";
                        }
                        else
                        {
                            ss << "((" << type << "*)(pool_base_ptr + "
                               << node->get_inputs()[5].get_tensor().get_pool_offset() << "))";
                            writer << "dyn_post_op_scales.assign(" << ss.str() << ", " << ss.str()
                                   << " + " << std::to_string(sum_scales_size) << ");\n";
                        }
                        writer << "ops.append_sum(dyn_post_op_scales[0]);\n";
                    }

                    if (has_relu<OP>(node))
                    {
                        writer << "const float ops_scale = 1.f;\n";
                        writer << "const float ops_alpha = -0.f; // relu negative slope\n";
                        writer << "const float ops_beta = 0.f;\n";
                        writer << "ops.append_eltwise("
                                  "ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, "
                                  "ops_beta);\n";
                    }

                    writer << "mkldnn::primitive_attr conv_attr;\n";
                    writer << "conv_attr.set_post_ops(ops);\n";

                    if (mkldnn_emitter.is_quantized_conv<OP>())
                    {
                        auto scale_index = mkldnn_emitter.get_scale_index<OP>();
                        auto c = std::dynamic_pointer_cast<ngraph::op::Constant>(
                            node->get_arguments()[scale_index]);
                        auto scales_size = shape_size(convolution->get_input_shape(scale_index));
                        const element::Type& et = node->get_input_element_type(scale_index);
                        std::string type = et.c_type_string();
                        std::stringstream ss;
                        if (c)
                        {
                            ss << "((" << type << "*)(" << c->get_data_ptr() << "))\n";
                        }
                        else
                        {
                            ss << "((" << type << "*)(pool_base_ptr + "
                               << node->get_inputs()[scale_index].get_tensor().get_pool_offset()
                               << "))";
                        }
                        writer << "std::vector<float> dyn_scales;\n";
                        writer << "dyn_scales.assign(" << ss.str() << ", " << ss.str() << " + "
                               << std::to_string(scales_size) << ");\n";
                        writer << "// use conv channelwise (dim 1, mask=2^1) if dyn_scales is a "
                                  "vector \n";
                        writer << "const int mask = " << std::to_string(scales_size)
                               << " == 1 ? 0 : 2;\n";
                        writer << "conv_attr.set_int_output_round_mode(mkldnn::round_mode::round_"
                                  "nearest);\n";
                        writer << "conv_attr.set_output_scales(mask, dyn_scales);\n";
                    }

                    //emit_memory_primitive_build(writer, desc_names, deps);
                    writer << "mkldnn::primitive* prim;\n";
                    if (mkldnn_emitter.has_bias<OP>())
                    {
                        writer << "prim = new mkldnn::convolution_forward({conv_desc, conv_attr, "
                                  "cg_ctx->global_cpu_engine},"
                                  "*cg_ctx->mkldnn_primitives["
                               << std::to_string(deps[0]) << "],\n"
                                                             "*cg_ctx->mkldnn_primitives["
                               << std::to_string(deps[1]) << "],\n"
                                                             "*cg_ctx->mkldnn_primitives["
                               << std::to_string(deps[2]) << "],\n"
                                                             "*cg_ctx->mkldnn_primitives["
                               << std::to_string(deps[3]) << "]);\n";
                    }
                    else
                    {
                        writer << "prim = new mkldnn::convolution_forward({conv_desc, conv_attr, "
                                  "cg_ctx->global_cpu_engine},"
                                  "*cg_ctx->mkldnn_primitives["
                               << std::to_string(deps[0]) << "],\n"
                                                             "*cg_ctx->mkldnn_primitives["
                               << std::to_string(deps[1]) << "],\n"
                                                             "*cg_ctx->mkldnn_primitives["
                               << std::to_string(deps[2]) << "]);\n";
                    }
                    writer << "cg_ctx->mkldnn_primitives[" << std::to_string(index)
                           << "] = prim;\n";
                    construct_string = writer.get_code();
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Convolution)
                {
                    construct_primitive_build_string_conv<Convolution>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolution)
                {
                    construct_primitive_build_string_conv<QuantizedConvolution>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void
                    MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(ConvolutionRelu)
                {
                    construct_primitive_build_string_conv<ConvolutionRelu>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionRelu)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionRelu>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void
                    MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(ConvolutionBias)
                {
                    construct_primitive_build_string_conv<ConvolutionBias>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionBias)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionBias>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    ConvolutionBiasAdd)
                {
                    construct_primitive_build_string_conv<ConvolutionBiasAdd>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionBiasAdd)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionBiasAdd>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(ConvolutionAdd)
                {
                    construct_primitive_build_string_conv<ConvolutionAdd>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionBiasSignedAdd)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionBiasSignedAdd>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    GroupConvolution)
                {
                    construct_primitive_build_string_conv<GroupConvolution>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    GroupConvolutionBias)
                {
                    construct_primitive_build_string_conv<GroupConvolutionBias>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <typename OpTy>
                size_t build_convolution_backward(MKLDNNEmitter& mkldnn_emitter,
                                                  const ngraph::Node* node)
                {
                    auto convolution = static_cast<const OpTy*>(node);

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto arg0_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto arg1_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto out0_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    if (std::is_same<OpTy, ngraph::op::ConvolutionBackpropData>())
                    {
                        // MKLDNN relies on named formats for kernel selection
                        if (arg0_desc.data.format == mkldnn_nchw)
                        {
                            arg0_desc.data.format = mkldnn_oihw;
                        }
                        if (arg0_desc.data.format == mkldnn_ncdhw)
                        {
                            arg0_desc.data.format = mkldnn_oidhw;
                        }

                        return mkldnn_emitter.build_convolution_backward_data(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }
                    if (std::is_same<OpTy, ngraph::op::ConvolutionBackpropFilters>())
                    {
                        return mkldnn_emitter.build_convolution_backward_weights(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }
                    if (std::is_same<OpTy, ngraph::op::ConvolutionBiasBackpropFiltersBias>())
                    {
                        auto out1_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);
                        return mkldnn_emitter.build_convolution_backward_weights_bias(
                            arg0_desc,
                            arg1_desc,
                            out0_desc,
                            out1_desc,
                            convolution->get_window_movement_strides_forward(),
                            window_dilation_strides_adjusted,
                            convolution->get_padding_below_forward(),
                            convolution->get_padding_above_forward());
                    }

                    throw ngraph_error(std::string("Unknown op ") + convolution->get_name());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBackpropFilters)
                {
                    return build_convolution_backward<ConvolutionBackpropFilters>(mkldnn_emitter,
                                                                                  node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBackpropData)
                {
                    return build_convolution_backward<ConvolutionBackpropData>(mkldnn_emitter,
                                                                               node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    ConvolutionBiasBackpropFiltersBias)
                {
                    return build_convolution_backward<ConvolutionBiasBackpropFiltersBias>(
                        mkldnn_emitter, node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPool)
                {
                    auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_pooling_forward(
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        max_pool->get_window_movement_strides(),
                        max_pool->get_window_shape(),
                        max_pool->get_padding_below(),
                        max_pool->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedMaxPool)
                {
                    return mkldnn_emitter.build_quantized_max_pool(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedAvgPool)
                {
                    return mkldnn_emitter.build_quantized_avg_pool(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolWithIndices)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto max_pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);

                    return mkldnn_emitter.build_max_pooling_with_indices_forward(
                        mkldnn::algorithm::pooling_max,
                        input_desc,
                        result_desc,
                        max_pool->get_window_movement_strides(),
                        max_pool->get_window_shape(),
                        max_pool->get_padding_below(),
                        max_pool->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(AvgPool)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);

                    return mkldnn_emitter.build_pooling_forward(
                        (avg_pool->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        input_desc,
                        result_desc,
                        avg_pool->get_window_movement_strides(),
                        avg_pool->get_window_shape(),
                        avg_pool->get_padding_below(),
                        avg_pool->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(AvgPoolBackprop)
                {
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);

                    return mkldnn_emitter.build_pooling_backward(
                        (apb->get_include_padding_in_avg_computation()
                             ? mkldnn::algorithm::pooling_avg_include_padding
                             : mkldnn::algorithm::pooling_avg_exclude_padding),
                        diff_dst_desc,
                        diff_src_desc,
                        apb->get_window_movement_strides(),
                        apb->get_window_shape(),
                        apb->get_padding_below(),
                        apb->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolBackprop)
                {
                    auto fprop_src_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);

                    return mkldnn_emitter.build_max_pooling_backward(
                        mkldnn::algorithm::pooling_max,
                        fprop_src_desc,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(MaxPoolWithIndicesBackprop)
                {
                    auto diff_dst_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto diff_src_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mpb = static_cast<const ngraph::op::MaxPoolWithIndicesBackprop*>(node);

                    return mkldnn_emitter.build_max_pooling_with_indices_backward(
                        mkldnn::algorithm::pooling_max,
                        diff_dst_desc,
                        diff_src_desc,
                        mpb->get_window_movement_strides(),
                        mpb->get_window_shape(),
                        mpb->get_padding_below(),
                        mpb->get_padding_above());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    ngraph::runtime::cpu::op::ConvertLayout)
                {
                    const auto& args = node->get_inputs();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    // This is a special case to handle nchw(oihw) to goihw/Goihw16g/Goihw8g for
                    // GroupConvolution's weights.
                    if (input_desc.data.format == mkldnn_nchw &&
                        result_desc.data.format == mkldnn_goihw)
                    {
                        input_desc = result_desc;
                    }
                    else if (input_desc.data.format == mkldnn_nchw &&
                             input_desc.data.ndims == 4 /*nchw*/ &&
                             result_desc.data.ndims == 5 /*Goihw16g/Goihw8g/etc*/ &&
                             node->get_users().size() == 1)
                    {
                        Shape weights_shape_groups;
                        if (auto gconv = std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(
                                node->get_users()[0]))
                        {
                            weights_shape_groups = gconv->get_weights_dimensions();
                        }
                        else if (auto gconvb =
                                     std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(
                                         node->get_users()[0]))
                        {
                            weights_shape_groups = gconvb->get_weights_dimensions();
                        }
                        else
                        {
                            throw ngraph_error(
                                "Incompatible input/output shape in ConvertLayout op");
                        }
                        input_desc = mkldnn::memory::desc(
                            mkldnn::memory::dims(weights_shape_groups.begin(),
                                                 weights_shape_groups.end()),
                            mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                            mkldnn::memory::format::goihw);
                    }

                    return mkldnn_emitter.build_reorder(input_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ReluBackprop)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_relu_backward(input_desc, delta_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Relu)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    return mkldnn_emitter.build_relu_forward(input_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(LeakyRelu)
                {
                    auto leaky_relu_node = static_cast<const ngraph::op::LeakyRelu*>(node);
                    float alpha = leaky_relu_node->get_alpha();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_leaky_relu(input_desc, result_desc, alpha);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BoundedRelu)
                {
                    auto bounded_relu_node = static_cast<const ngraph::op::BoundedRelu*>(node);
                    float alpha = bounded_relu_node->get_alpha();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_bounded_relu(input_desc, result_desc, alpha);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Sigmoid)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    return mkldnn_emitter.build_sigmoid_forward(input_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(SigmoidBackprop)
                {
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_sigmoid_backward(
                        input_desc, delta_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Softmax)
                {
                    auto softmax = static_cast<const ngraph::op::Softmax*>(node);

                    if (softmax->get_axes().size() != 1)
                    {
                        throw ngraph_error("MKLDNN supports softmax only across single axis");
                    }

                    int softmax_axis = static_cast<int>(*(softmax->get_axes().begin()));
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    return mkldnn_emitter.build_softmax_forward(
                        input_desc, result_desc, softmax_axis);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Dequantize)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    return mkldnn_emitter.build_dequantization(node, input_data_desc, result_desc);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Quantize)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto quantize = static_cast<const ngraph::op::Quantize*>(node);
                    auto scale_const_op =
                        std::dynamic_pointer_cast<Constant>(quantize->get_argument(1));

                    if (scale_const_op == nullptr)
                    {
                        throw ngraph_error("Quantize scale must be a constant");
                    }

                    auto scale = scale_const_op->get_vector<float>();
                    std::vector<float> scales;
                    scales.push_back(1.0 / scale[0]);

                    return mkldnn_emitter.build_quantize_reorder(
                        input_data_desc, result_desc, scales);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConcat)
                {
                    int args_size = node->get_inputs().size();

                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0; i < args_size; i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim =
                        (static_cast<const QuantizedConcat*>(node))->get_concatenation_axis();
                    return mkldnn_emitter.build_concat(inputs_data_desc, result_desc, concat_dim);
                }
            }
        }
    }
}

using namespace ngraph::runtime::cpu::pass;

#define TI(x) std::type_index(typeid(x))

static const PrimitiveBuildOpMap prim_build_dispatcher{
    {TI(Convert), &MKLDNNPrimitiveBuildPass::build_primitive<Convert>},
    {TI(runtime::cpu::op::ConvertLayout),
     &MKLDNNPrimitiveBuildPass::build_primitive<runtime::cpu::op::ConvertLayout>},
    {TI(AvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPool>},
    {TI(AvgPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPoolBackprop>},
    {TI(BoundedRelu), &MKLDNNPrimitiveBuildPass::build_primitive<BoundedRelu>},
    {TI(ConvolutionBackpropData),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropData>},
    {TI(ConvolutionBackpropFilters),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropFilters>},
    {TI(MaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPool>},
    {TI(MaxPoolWithIndices), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndices>},
    {TI(MaxPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolBackprop>},
    {TI(MaxPoolWithIndicesBackprop),
     &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndicesBackprop>},
    {TI(ConvolutionBiasBackpropFiltersBias),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBiasBackpropFiltersBias>},
    {TI(Relu), &MKLDNNPrimitiveBuildPass::build_primitive<Relu>},
    {TI(ReluBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<ReluBackprop>},
    {TI(LeakyRelu), &MKLDNNPrimitiveBuildPass::build_primitive<LeakyRelu>},
    {TI(Sigmoid), &MKLDNNPrimitiveBuildPass::build_primitive<Sigmoid>},
    {TI(SigmoidBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<SigmoidBackprop>},
    {TI(QuantizedMaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedMaxPool>},
    {TI(QuantizedAvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedAvgPool>},
    {TI(Softmax), &MKLDNNPrimitiveBuildPass::build_primitive<Softmax>},
    {TI(Slice), &MKLDNNPrimitiveBuildPass::build_primitive<Slice>},
    {TI(ReplaceSlice), &MKLDNNPrimitiveBuildPass::build_primitive<ReplaceSlice>},
    {TI(UpdateSlice), &MKLDNNPrimitiveBuildPass::build_primitive<UpdateSlice>},
    {TI(Quantize), &MKLDNNPrimitiveBuildPass::build_primitive<Quantize>},
    {TI(Dequantize), &MKLDNNPrimitiveBuildPass::build_primitive<Dequantize>},
    {TI(QuantizedConcat), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConcat>},
    {TI(GetOutputElement), &MKLDNNPrimitiveBuildPass::build_primitive<GetOutputElement>},
};

static const PrimitiveBuildStringConstructOpMap prim_build_string_construct_dispatcher{
    {TI(Add), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Add>},
    {TI(Concat), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Concat>},
    {TI(BatchNormInference),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<BatchNormInference>},
    {TI(BatchNormTraining),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<BatchNormTraining>},
    {TI(BatchNormInferenceRelu),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<BatchNormInferenceRelu>},
    {TI(BatchNormTrainingRelu),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<BatchNormTrainingRelu>},
    {TI(BatchNormTrainingBackprop),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<BatchNormTrainingBackprop>},
    {TI(LRN), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<LRN>},
    {TI(Lstm), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Lstm>},
    {TI(Rnn), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Rnn>},
    {TI(Convolution), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Convolution>},
    {TI(ConvolutionRelu),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<ConvolutionRelu>},
    {TI(ConvolutionBias),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<ConvolutionBias>},
    {TI(ConvolutionBiasAdd),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<ConvolutionBiasAdd>},
    {TI(ConvolutionAdd),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<ConvolutionAdd>},
    {TI(GroupConvolution),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<GroupConvolution>},
    {TI(GroupConvolutionBias),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<GroupConvolutionBias>},
    {TI(QuantizedConvolution),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolution>},
    {TI(QuantizedConvolutionRelu),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolutionRelu>},
    {TI(QuantizedConvolutionBias),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolutionBias>},
    {TI(QuantizedConvolutionBiasAdd),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolutionBiasAdd>},
    {TI(QuantizedConvolutionBiasSignedAdd),
     &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<
         QuantizedConvolutionBiasSignedAdd>},
};

// Check if the node builds primitives at first iteration.
// Needed during transition when there are two maps.
static bool in_new_map(const std::shared_ptr<Node>& node)
{
    if (std::dynamic_pointer_cast<ngraph::op::Add>(node) ||
        std::dynamic_pointer_cast<ngraph::op::Concat>(node) ||
        std::dynamic_pointer_cast<ngraph::op::BatchNormInference>(node) ||
        std::dynamic_pointer_cast<ngraph::op::BatchNormTraining>(node) ||
        std::dynamic_pointer_cast<ngraph::op::BatchNormInferenceRelu>(node) ||
        std::dynamic_pointer_cast<ngraph::op::BatchNormTrainingRelu>(node) ||
        std::dynamic_pointer_cast<ngraph::op::BatchNormTrainingBackprop>(node) ||
        std::dynamic_pointer_cast<ngraph::op::LRN>(node) ||
        std::dynamic_pointer_cast<ngraph::op::Lstm>(node) ||
        std::dynamic_pointer_cast<ngraph::op::Rnn>(node) ||
        std::dynamic_pointer_cast<ngraph::op::Convolution>(node) ||
        std::dynamic_pointer_cast<ngraph::op::ConvolutionRelu>(node) ||
        std::dynamic_pointer_cast<ngraph::op::ConvolutionBias>(node) ||
        std::dynamic_pointer_cast<ngraph::op::ConvolutionBiasAdd>(node) ||
        std::dynamic_pointer_cast<ngraph::op::ConvolutionAdd>(node) ||
        std::dynamic_pointer_cast<ngraph::op::QuantizedConvolution>(node) ||
        std::dynamic_pointer_cast<ngraph::op::QuantizedConvolutionRelu>(node) ||
        std::dynamic_pointer_cast<ngraph::op::QuantizedConvolutionBias>(node) ||
        std::dynamic_pointer_cast<ngraph::op::QuantizedConvolutionBiasAdd>(node) ||
        std::dynamic_pointer_cast<ngraph::op::QuantizedConvolutionBiasSignedAdd>(node) ||
        std::dynamic_pointer_cast<ngraph::op::GroupConvolution>(node) ||
        std::dynamic_pointer_cast<ngraph::op::GroupConvolutionBias>(node))
    {
        return true;
    }
    return false;
}

bool MKLDNNPrimitiveBuildPass::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& shp_node : nodes)
    {
        if (in_new_map(shp_node))
        {
            continue;
        }

        Node* node = shp_node.get();

        if (mkldnn_utils::use_mkldnn_kernel(node))
        {
            auto handler = prim_build_dispatcher.find(TI(*node));
            NGRAPH_CHECK(handler != prim_build_dispatcher.end(),
                         "Unsupported node '",
                         node->description(),
                         "' in MKLDNNPrimitiveBuildPass");

            size_t primitive_idx = handler->second(m_mkldnn_emitter, node);
            m_node_primitive_idx_map[node] = primitive_idx;
        }
    }

    std::ofstream desc_file(m_desc_filename, std::ios::out | std::ios::binary);
    for (const auto& shp_node : nodes)
    {
        if (!in_new_map(shp_node))
        {
            continue;
        }

        Node* node = shp_node.get();

        if (mkldnn_utils::use_mkldnn_kernel(node))
        {
            auto handler = prim_build_string_construct_dispatcher.find(TI(*node));
            NGRAPH_CHECK(handler != prim_build_string_construct_dispatcher.end(),
                         "Unsupported node '",
                         node->description(),
                         "' in MKLDNNPrimitiveBuildPass");

            std::string construct_string;
            std::vector<size_t> deps;
            size_t index;
            handler->second(m_mkldnn_emitter, node, construct_string, deps, index, desc_file);
            m_node_primitive_string_deps_index_map[node] =
                std::tuple<std::string, std::vector<size_t>, size_t>(construct_string, deps, index);
        }
    }

    return false;
}

#undef TI
