//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "cpu_dnnl_primitive_build.hpp"

#include "ngraph/code_writer.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_wrapper.hpp"
#include "ngraph/runtime/cpu/dnnl_emitter.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"

#define WRITE_DNNL_DIMS(X) writer << "dnnl::memory::dims{" << join(X) << "}, \n";

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
                                                   std::vector<dnnl::memory::desc>& descs,
                                                   size_t index)
                {
                    for (size_t i = 0; i < descs.size(); i++)
                    {
                        desc_file << index;
                        desc_file.write(reinterpret_cast<char*>(&descs[i]),
                                        sizeof(dnnl::memory::desc));
                        index++;
                    }
                }

                // The following functions build the DNNL primitive for each type of nGraph Node.

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Add)
                {
                    auto input0_data_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto input1_data_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto sum_pd = dnnl_emitter.get_elementwise_add_desc(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_sum(sum_pd);

                    // Add needs 4 primitives: input0, input1, result, and sum.
                    index = dnnl_emitter.reserve_primitive_space(4);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {
                        input0_data_desc, input1_data_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "std::vector<float> scale_vector(2, 1);\n";
                    writer << "std::vector<dnnl::memory::desc> inputs_desc{"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], "
                           << "*cg_ctx->dnnl_descriptors[" << desc_index + 1 << "]};\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    // elementwise sum primitive descriptor
                    writer << "dnnl::sum::primitive_desc sum_pd = "
                              "dnnl::sum::primitive_desc(*cg_ctx->dnnl_descriptors["
                           << desc_index + 2
                           << "], "
                              "scale_vector, inputs_desc, cg_ctx->global_cpu_engine, attr);\n";

                    writer << "\n// build sum primitive\n";

                    // sum primitive
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::sum(sum_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(sum_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <typename OP>
                void construct_primitive_build_string_rnn(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file)
                {
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

                    auto get_dnnl_rnn_direction_string = [&]() {
                        switch (direction)
                        {
                        case 1:
                            return std::string("dnnl::rnn_direction::unidirectional_left2right");
                        case 2: return std::string("dnnl::rnn_direction::bidirectional_concat");
                        default: throw ngraph_error("unsupported dnnl rnn direction");
                        }
                    };

                    auto get_dnnl_rnn_direction = [&]() {
                        switch (direction)
                        {
                        case 1: return dnnl::rnn_direction::unidirectional_left2right;
                        case 2: return dnnl::rnn_direction::bidirectional_concat;
                        default: throw ngraph_error("unsupported dnnl rnn direction");
                        }
                    };

                    if (node->get_output_shape(0).size() == 2 &&
                        (node->get_output_shape(0)[1] != direction * feature_size))
                    {
                        throw ngraph_error(
                            "input slc{ht} feature size is not equal to output dlc{ht} feature "
                            "size ");
                    }

                    if (node->get_output_shape(1).size() == 2 &&
                        (node->get_output_shape(1)[1] != feature_size) &&
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
                    Shape src_iter_tz{num_fused_layers, direction, batch, feature_size};
                    Shape src_iter_c_tz{num_fused_layers, direction, batch, feature_size};
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
                    Shape dst_iter_tz{num_fused_layers, direction, batch, feature_size};
                    Shape dst_iter_c_tz{num_fused_layers, direction, batch, feature_size};

                    // We create the memory descriptors used by the user
                    auto src_layer_md =
                        dnnl_emitter.build_memory_descriptor(src_layer_tz,
                                                             node->get_input_element_type(0),
                                                             dnnl::memory::format_tag::tnc);
                    auto src_iter_md =
                        dnnl_emitter.build_memory_descriptor(src_iter_tz,
                                                             node->get_input_element_type(1),
                                                             dnnl::memory::format_tag::ldnc);
                    auto src_iter_c_md =
                        dnnl_emitter.build_memory_descriptor(src_iter_c_tz,
                                                             node->get_input_element_type(1),
                                                             dnnl::memory::format_tag::ldnc);
                    auto wei_layer_md =
                        dnnl_emitter.build_memory_descriptor(wei_layer_tz,
                                                             node->get_input_element_type(2),
                                                             dnnl::memory::format_tag::ldigo);
                    auto wei_iter_md =
                        dnnl_emitter.build_memory_descriptor(wei_iter_tz,
                                                             node->get_input_element_type(3),
                                                             dnnl::memory::format_tag::ldigo);
                    auto bias_md = dnnl_emitter.build_memory_descriptor(
                        bias_tz, node->get_input_element_type(4), dnnl::memory::format_tag::ldgo);
                    auto dst_layer_md =
                        dnnl_emitter.build_memory_descriptor(dst_layer_tz,
                                                             node->get_output_element_type(0),
                                                             dnnl::memory::format_tag::tnc);
                    auto dst_iter_md =
                        dnnl_emitter.build_memory_descriptor(dst_iter_tz,
                                                             node->get_output_element_type(1),
                                                             dnnl::memory::format_tag::ldnc);
                    auto dst_iter_c_md =
                        dnnl_emitter.build_memory_descriptor(dst_iter_c_tz,
                                                             node->get_output_element_type(1),
                                                             dnnl::memory::format_tag::ldnc);

                    // query scratchpad size
                    auto rnn_desc = dnnl::lstm_forward::desc(dnnl::prop_kind::forward_training,
                                                             get_dnnl_rnn_direction(),
                                                             src_layer_md,
                                                             src_iter_md,
                                                             src_iter_c_md,
                                                             wei_layer_md,
                                                             wei_iter_md,
                                                             bias_md,
                                                             dst_layer_md,
                                                             dst_iter_md,
                                                             dst_iter_c_md);
                    scratchpad_size = dnnl_emitter.query_scratchpad_rnn_forward(rnn_desc);

                    // Lstm/Rnn needs 11 primitives: src_layer, src_iter, src_iter_c, weights_layer,
                    // weights_iter, bias,
                    // dst_layer, dst_iter, dst_iter_c, workspace, and rnn_forward.
                    // It needs a new workspace.
                    index = dnnl_emitter.reserve_primitive_space(
                        11, false /* fwd and bwd */, true /* new workspace */);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {src_layer_md,
                                                             src_iter_md,
                                                             src_iter_c_md,
                                                             wei_layer_md,
                                                             wei_iter_md,
                                                             bias_md,
                                                             dst_layer_md,
                                                             dst_iter_md,
                                                             dst_iter_c_md};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "\n// build lstm/rnn primitive descriptor\n";
                    writer << "auto rnn_desc = "
                              "dnnl::lstm_forward::desc(dnnl::prop_kind::forward_training, "
                           << get_dnnl_rnn_direction_string()
                           << ", "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 2
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 3
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 4
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 5
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 6
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 7
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 8 << "]);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";
                    writer << "auto rnn_prim_desc = dnnl::lstm_forward::primitive_desc(rnn_desc, "
                              "attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "cg_ctx->dnnl_memories[" << std::to_string(deps[9])
                           << "] = new "
                              "dnnl::memory(rnn_prim_desc.workspace_desc(), "
                              "cg_ctx->global_cpu_engine, nullptr);\n";
                    writer << "auto workspace = "
                              "(char*)malloc(rnn_prim_desc.workspace_desc().get_size());"
                              "\n";
                    writer << "if (!workspace)\n";
                    writer.block_begin();
                    writer << "throw std::bad_alloc();\n";
                    writer.block_end();
                    writer << "cg_ctx->dnnl_workspaces.push_back(workspace);\n";

                    deps[10] = dnnl_emitter.reserve_workspace();

                    writer << "\n// build lstm/rnn primitive\n";
                    // lstm/rnn primitive
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::lstm_forward(rnn_prim_desc);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(rnn_prim_desc.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Lstm)
                {
                    construct_primitive_build_string_rnn<Lstm>(dnnl_emitter,
                                                               node,
                                                               construct_string,
                                                               deps,
                                                               index,
                                                               scratchpad_size,
                                                               desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Rnn)
                {
                    construct_primitive_build_string_rnn<Rnn>(dnnl_emitter,
                                                              node,
                                                              construct_string,
                                                              deps,
                                                              index,
                                                              scratchpad_size,
                                                              desc_file);
                }

                template <typename OP>
                void construct_primitive_build_string_batchnorm(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file,
                    const bool append_relu,
                    const bool training)
                {
                    // batchnorm forward needs 6 primitives: input, weights, result, mean,
                    // variance, and batch_normalization_forward.
                    index = dnnl_emitter.reserve_primitive_space(6);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    if (append_relu)
                    {
                        writer << "dnnl::post_ops pops;\n";
                        writer << "const float ops_scale = 1.f;\n";
                        writer << "const float ops_alpha = -0.f; // relu negative slope\n";
                        writer << "const float ops_beta = 0.f;\n";

                        writer << "pops.append_eltwise("
                                  "ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, "
                                  "ops_beta);\n";
                    }
                    else
                    {
                        writer << "dnnl::post_ops pops = dnnl::post_ops();\n";
                    }

                    auto weights_shape =
                        Shape{2, node->get_input_tensor(0).get_tensor_layout()->get_size()};
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                    auto weights_desc =
                        dnnl_emitter.build_memory_descriptor(weights_shape,
                                                             node->get_input_element_type(0),
                                                             dnnl::memory::format_tag::nc);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    const float ops_scale = 1.f;
                    const float ops_alpha = -0.f; // relu negative slope
                    const float ops_beta = 0.f;

                    dnnl::post_ops ops;
                    if (append_relu)
                    {
                        ops.append_eltwise(
                            ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    bool use_global_stats;
                    const dnnl::memory::desc *mean_desc, *variance_desc;
                    if (training && node->get_input_size() == 3)
                    {
                        mean_desc = &dnnl_utils::get_output_dnnl_md(node, 1);
                        variance_desc = &dnnl_utils::get_output_dnnl_md(node, 2);
                        use_global_stats = false;
                        // query scratchpad size
                        auto batchnorm_desc =
                            dnnl_emitter.get_batchnorm_forward_desc<OP>(node, true);
                        scratchpad_size =
                            dnnl_emitter.query_scratchpad_batchnorm_forward(batchnorm_desc, ops);
                    }
                    else
                    {
                        mean_desc = &dnnl_utils::get_input_dnnl_md(node, 3);
                        variance_desc = &dnnl_utils::get_input_dnnl_md(node, 4);
                        use_global_stats = true;
                        // query scratchpad size
                        auto batchnorm_desc =
                            dnnl_emitter.get_batchnorm_forward_desc<OP>(node, false);
                        scratchpad_size =
                            dnnl_emitter.query_scratchpad_batchnorm_forward(batchnorm_desc, ops);
                    }

                    auto batchnorm = static_cast<const OP*>(node);
                    auto eps = batchnorm->get_eps_value();

                    writer << "dnnl::primitive_attr bn_attr;\n";
                    writer << "bn_attr.set_post_ops(pops);\n";
                    writer << "bn_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build batchnorm primitive descriptor\n";
                    if (use_global_stats)
                    {
                        // Write memory descriptors to file
                        std::vector<dnnl::memory::desc> descs = {
                            input_desc, *mean_desc, *variance_desc, weights_desc, result_desc};
                        auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                        dnnl_emitter.reserve_descriptor_space(descs.size());
                        serialize_memory_descs(desc_file, descs, deps[0]);

                        writer << "auto batchnorm_desc = "
                                  "dnnl::batch_normalization_forward::desc(dnnl::prop_kind::"
                                  "forward_training, "
                                  "*cg_ctx->dnnl_descriptors["
                               << desc_index << "], " << eps
                               << ", "
                                  "dnnl::normalization_flags::use_scale_shift | "
                                  "dnnl::normalization_flags::use_global_stats);\n";
                    }
                    else
                    {
                        // Write memory descriptors to file
                        std::vector<dnnl::memory::desc> descs = {
                            input_desc, weights_desc, result_desc, *mean_desc, *variance_desc};
                        auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                        dnnl_emitter.reserve_descriptor_space(descs.size());
                        serialize_memory_descs(desc_file, descs, deps[0]);

                        writer << "auto batchnorm_desc = "
                                  "dnnl::batch_normalization_forward::desc(dnnl::prop_kind::"
                                  "forward_training, "
                                  "*cg_ctx->dnnl_descriptors["
                               << desc_index << "], " << eps
                               << ", "
                                  "dnnl::normalization_flags::use_scale_shift);\n";
                    }
                    writer << "auto batchnorm_prim_desc = "
                              "dnnl::batch_normalization_forward::primitive_desc(batchnorm_"
                              "desc, "
                              "bn_attr, cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build batchnorm primitive\n";

                    // batchnorm primitive
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new "
                              "dnnl::batch_normalization_forward(batchnorm_prim_desc);\n";
                    writer
                        << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                        << "] = new dnnl::memory::desc(batchnorm_prim_desc.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void
                    DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(BatchNormTraining)
                {
                    construct_primitive_build_string_batchnorm<BatchNormTraining>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file,
                        false /*Append relu*/,
                        true /*Training*/);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormInference)
                {
                    construct_primitive_build_string_batchnorm<BatchNormInference>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file,
                        false /*Append relu*/,
                        false /*Training*/);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormTrainingRelu)
                {
                    construct_primitive_build_string_batchnorm<BatchNormTrainingRelu>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file,
                        true /*Append relu*/,
                        true /*Training*/);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormInferenceRelu)
                {
                    construct_primitive_build_string_batchnorm<BatchNormInferenceRelu>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file,
                        true /*Append relu*/,
                        false /*Training*/);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    BatchNormTrainingBackprop)
                {
                    const auto* batchnorm = static_cast<const BatchNormTrainingBackprop*>(node);
                    auto eps = batchnorm->get_eps_value();

                    auto weights_shape =
                        Shape{2, node->get_input_tensor(0).get_tensor_layout()->get_size()};
                    auto weights_desc =
                        dnnl_emitter.build_memory_descriptor(weights_shape,
                                                             node->get_input_element_type(0),
                                                             dnnl::memory::format_tag::nc);
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                    auto mean_desc = dnnl_utils::get_input_dnnl_md(node, 3);
                    auto variance_desc = dnnl_utils::get_input_dnnl_md(node, 4);
                    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 5);
                    auto dinput_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    auto dweights_desc =
                        dnnl_emitter.build_memory_descriptor(weights_shape,
                                                             node->get_input_element_type(0),
                                                             dnnl::memory::format_tag::nc);

                    // query scratchpad size
                    auto batchnorm_desc = dnnl_emitter.get_batchnorm_backward_desc(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_batchnorm_backward(
                        batchnorm_desc, input_desc, eps);

                    // batchnorm backward needs 8 primitives: weights, input, mean, variance,
                    // dinput, dweights, and batch_normalization_backward.
                    index = dnnl_emitter.reserve_primitive_space(8);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {weights_desc,
                                                             input_desc,
                                                             mean_desc,
                                                             variance_desc,
                                                             delta_desc,
                                                             dinput_desc,
                                                             dweights_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto batchnorm_fdesc = "
                              "dnnl::batch_normalization_forward::desc(dnnl::prop_kind::"
                              "forward_training, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "], " << eps
                           << ", "
                              "dnnl::normalization_flags::use_scale_shift);\n";
                    writer << "auto batchnorm_fpd = "
                              "dnnl::batch_normalization_forward::primitive_desc("
                              "batchnorm_fdesc, cg_ctx->global_cpu_engine);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "auto batchnorm_desc = "
                              "dnnl::batch_normalization_backward::desc(dnnl::prop_kind::"
                              "backward, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 4
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "], " << eps
                           << ", "
                              "dnnl::normalization_flags::use_scale_shift);\n";
                    writer << "auto batchnorm_prim_desc = "
                              "dnnl::batch_normalization_backward::primitive_desc(batchnorm_"
                              "desc, "
                              "attr, cg_ctx->global_cpu_engine, batchnorm_fpd);\n";

                    writer << "\n// build batchnorm primitive\n";

                    // batchnorm primitive
                    writer << "\n// build batchnorm primitives\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new "
                              "dnnl::batch_normalization_backward(batchnorm_prim_desc);\n";
                    writer
                        << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                        << "] = new dnnl::memory::desc(batchnorm_prim_desc.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <typename OP>
                void construct_primitive_build_string_concat(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file)
                {
                    auto concat = static_cast<OP*>(node);
                    size_t concat_dim = concat->get_concatenation_axis();
                    size_t nargs = node->get_input_size();

                    // query scratchpad size
                    auto concat_pd = dnnl_emitter.get_concat_desc<OP>(node, nargs);
                    scratchpad_size = dnnl_emitter.query_scratchpad_concat(concat_pd);

                    // Concat needs number of inputs plus 2 primitives; those two are for result and
                    // concat.
                    index = dnnl_emitter.reserve_primitive_space(nargs + 2);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs;
                    for (size_t i = 0; i < nargs; i++)
                    {
                        descs.push_back(dnnl_utils::get_input_dnnl_md(node, i));
                    }
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    descs.push_back(result_desc);
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "std::vector<dnnl::memory::desc> inputs_desc;\n";

                    writer << "for (size_t i = " << desc_index << "; i < " << desc_index + nargs
                           << "; i++)\n";
                    writer.block_begin();
                    writer << "inputs_desc.push_back(*cg_ctx->dnnl_descriptors[i]);\n";
                    writer.block_end();

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "auto concat_prim_desc = "
                              "dnnl::concat::primitive_desc( "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + nargs << "], "
                           << std::to_string(static_cast<int>(concat_dim))
                           << ", inputs_desc, cg_ctx->global_cpu_engine, attr);\n";

                    writer << "\n// build concat primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::concat(concat_prim_desc);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(concat_prim_desc.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Concat)
                {
                    construct_primitive_build_string_concat<Concat>(dnnl_emitter,
                                                                    node,
                                                                    construct_string,
                                                                    deps,
                                                                    index,
                                                                    scratchpad_size,
                                                                    desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(LRN)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto lrn_desc = dnnl_emitter.get_lrn_forward_desc(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_lrn_forward(lrn_desc);

                    // LRN needs 3 primitives: input, result, and lrn_forward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    const auto* lrn = static_cast<const LRN*>(node);
                    auto alpha = static_cast<float>(lrn->get_alpha());
                    auto beta = static_cast<float>(lrn->get_beta());
                    auto bias = static_cast<float>(lrn->get_bias());
                    auto nsize = static_cast<int>(lrn->get_nsize());

                    writer << "auto lrn_desc = "
                              "dnnl::lrn_forward::desc(dnnl::prop_kind::forward_scoring, "
                              "dnnl::algorithm::lrn_across_channels, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], " << nsize << ", " << alpha << ", " << beta << ", "
                           << bias << ");\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "auto lrn_prim_desc = "
                              "dnnl::lrn_forward::primitive_desc(lrn_desc, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build lrn primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::lrn_forward(lrn_prim_desc);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(lrn_prim_desc.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Slice)
                {
                    const Slice* slice = static_cast<const Slice*>(node);
                    auto result_shape = node->get_output_shape(0);
                    auto lower_bounds = slice->get_lower_bounds();
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    scratchpad_size = dnnl_emitter.query_scratchpad_slice(
                        input_desc, result_desc, lower_bounds, result_shape);

                    // sub memory desc
                    auto dims = dnnl::memory::dims(result_shape.begin(), result_shape.end());
                    auto offsets = dnnl::memory::dims(lower_bounds.begin(), lower_bounds.end());
                    auto input_sub_desc = input_desc.submemory_desc(dims, offsets);

                    // Slice needs 3 primitives: input, result, and reorder.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_sub_desc, result_desc};
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build reorder primitives\n";
                    writer << "auto reorder_pd = "
                              "dnnl::reorder::primitive_desc("
                              "*cg_ctx->dnnl_memories["
                           << std::to_string(deps[0])
                           << "]"
                              ", *cg_ctx->dnnl_memories["
                           << std::to_string(deps[1]) << "], attr);\n";

                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::reorder(reorder_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(reorder_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <typename OP>
                void construct_primitive_build_string_conv(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file)
                {
                    auto convolution = static_cast<const OP*>(node);

                    // query scratchpad size
                    auto conv_desc = dnnl_emitter.get_convolution_forward_desc<OP>(node);
                    auto conv_attr = dnnl_emitter.get_convolution_forward_attr<OP>(node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_convolution_forward(conv_desc, conv_attr);

                    Strides window_dilation_strides_adjusted;

                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto data_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    auto strides = convolution->get_window_movement_strides();
                    auto pad_below = convolution->get_padding_below();
                    auto pad_above = convolution->get_padding_above();

                    if (dnnl_emitter.has_bias<OP>())
                    {
                        index = dnnl_emitter.reserve_primitive_space(5);
                    }
                    else
                    {
                        index = dnnl_emitter.reserve_primitive_space(4);
                    }
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    writer << "// Write in memory descriptors\n";
                    std::vector<dnnl::memory::desc> descs = {data_desc, weights_desc, result_desc};

                    if (dnnl_emitter.has_bias<OP>())
                    {
                        auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                        descs.insert(descs.begin() + 2, bias_desc);
                    }

                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "\n// build QConv primitive descriptor\n";
                    writer << "auto conv_desc = "
                              "dnnl::convolution_forward::desc(dnnl::prop_kind::forward,\n"
                              "dnnl::algorithm::convolution_direct,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "],\n";
                    if (dnnl_emitter.has_bias<OP>())
                    {
                        writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 2 << "],\n";
                    }
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + (descs.size() - 1)
                           << "],\n";
                    WRITE_DNNL_DIMS(strides);
                    WRITE_DNNL_DIMS(window_dilation_strides_adjusted);
                    WRITE_DNNL_DIMS(pad_below);
                    writer << "dnnl::memory::dims{" << join(pad_above) << "});\n";

                    writer << "dnnl::post_ops ops;\n";
                    if (std::is_same<OP, ngraph::op::v0::ConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::ConvolutionAdd>())
                    {
                        writer << "ops.append_sum(1.f);\n";
                    }

                    if (std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasAdd>() ||
                        std::is_same<OP, ngraph::op::v0::QuantizedConvolutionBiasSignedAdd>())
                    {
                        writer << "ops.append_sum(dyn_post_op_scales[0]);\n";
                    }

                    if (has_relu<OP>(node))
                    {
                        writer << "const float ops_scale = 1.f;\n";
                        writer << "const float ops_alpha = -0.f; // relu negative slope\n";
                        writer << "const float ops_beta = 0.f;\n";
                        writer << "ops.append_eltwise("
                                  "ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, "
                                  "ops_beta);\n";
                    }

                    writer << "dnnl::primitive_attr conv_attr;\n";
                    writer << "conv_attr.set_post_ops(ops);\n";
                    writer << "conv_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    if (dnnl_emitter.is_quantized_conv<OP>())
                    {
                        writer << "conv_attr.set_output_scales(mask, dyn_scales);\n";
                    }

                    writer << "auto conv_pd = dnnl::convolution_forward::primitive_desc("
                              "conv_desc, conv_attr, "
                              "cg_ctx->global_cpu_engine);\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::convolution_forward(conv_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(conv_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Convolution)
                {
                    construct_primitive_build_string_conv<Convolution>(dnnl_emitter,
                                                                       node,
                                                                       construct_string,
                                                                       deps,
                                                                       index,
                                                                       scratchpad_size,
                                                                       desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolution)
                {
                    construct_primitive_build_string_conv<QuantizedConvolution>(dnnl_emitter,
                                                                                node,
                                                                                construct_string,
                                                                                deps,
                                                                                index,
                                                                                scratchpad_size,
                                                                                desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(ConvolutionRelu)
                {
                    construct_primitive_build_string_conv<ConvolutionRelu>(dnnl_emitter,
                                                                           node,
                                                                           construct_string,
                                                                           deps,
                                                                           index,
                                                                           scratchpad_size,
                                                                           desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionRelu)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionRelu>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(ConvolutionBias)
                {
                    construct_primitive_build_string_conv<ConvolutionBias>(dnnl_emitter,
                                                                           node,
                                                                           construct_string,
                                                                           deps,
                                                                           index,
                                                                           scratchpad_size,
                                                                           desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionBias)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionBias>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    ConvolutionBiasAdd)
                {
                    construct_primitive_build_string_conv<ConvolutionBiasAdd>(dnnl_emitter,
                                                                              node,
                                                                              construct_string,
                                                                              deps,
                                                                              index,
                                                                              scratchpad_size,
                                                                              desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionBiasAdd)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionBiasAdd>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(ConvolutionAdd)
                {
                    construct_primitive_build_string_conv<ConvolutionAdd>(dnnl_emitter,
                                                                          node,
                                                                          construct_string,
                                                                          deps,
                                                                          index,
                                                                          scratchpad_size,
                                                                          desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    QuantizedConvolutionBiasSignedAdd)
                {
                    construct_primitive_build_string_conv<QuantizedConvolutionBiasSignedAdd>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(GroupConvolution)
                {
                    construct_primitive_build_string_conv<GroupConvolution>(dnnl_emitter,
                                                                            node,
                                                                            construct_string,
                                                                            deps,
                                                                            index,
                                                                            scratchpad_size,
                                                                            desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    GroupConvolutionBias)
                {
                    construct_primitive_build_string_conv<GroupConvolutionBias>(dnnl_emitter,
                                                                                node,
                                                                                construct_string,
                                                                                deps,
                                                                                index,
                                                                                scratchpad_size,
                                                                                desc_file);
                }

                template <typename OP>
                void construct_primitive_build_string_conv_backward_filters(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file)
                {
                    auto has_bias = false;
                    if (dnnl_emitter.has_bias<OP>())
                    {
                        has_bias = true;
                    }
                    auto convolution = static_cast<const OP*>(node);

                    // query scratchpad size
                    auto bwd_desc = dnnl_emitter.get_convolution_backward_weights_desc<OP>(node);
                    auto fwd_desc =
                        dnnl_emitter.get_convolution_forward_desc_for_backward_op<OP>(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_convolution_backward_weights(
                        fwd_desc, bwd_desc);

                    Strides window_dilation_strides_adjusted;
                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto arg0_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto arg1_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto out0_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    auto strides = convolution->get_window_movement_strides_forward();
                    auto pad_below = convolution->get_padding_below_forward();
                    auto pad_above = convolution->get_padding_above_forward();
                    dnnl::algorithm conv_algo = dnnl_utils::get_conv_algo();
                    auto conv_algo_string = conv_algo == dnnl::algorithm::convolution_auto
                                                ? "dnnl::algorithm::convolution_auto,\n"
                                                : "dnnl::algorithm::convolution_direct,\n";

                    std::vector<dnnl::memory::desc> descs = {arg0_desc, arg1_desc, out0_desc};
                    // ConvolutionBackpropFilter needs 4 primitives: src, diff_dst, diff_weights,
                    // and convolution_backward_weights.
                    // ConvolutionBackpropFiltersBias needs 5 primitives: src, diff_dst,
                    // diff_weights,
                    // diff_bias, and convolution_backward_weights.
                    if (has_bias)
                    {
                        index = dnnl_emitter.reserve_primitive_space(5);
                        auto out1_desc = dnnl_utils::get_output_dnnl_md(node, 1);
                        descs.push_back(out1_desc);
                    }
                    else
                    {
                        index = dnnl_emitter.reserve_primitive_space(4);
                    }
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto fwd_desc = "
                              "dnnl::convolution_forward::desc(dnnl::prop_kind::forward,\n";
                    writer << conv_algo_string;
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 2 << "],\n";
                    if (has_bias)
                    {
                        writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 3 << "],\n";
                    }
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(strides);
                    WRITE_DNNL_DIMS(window_dilation_strides_adjusted);
                    WRITE_DNNL_DIMS(pad_below);
                    writer << "dnnl::memory::dims{" << join(pad_above) << "});\n";

                    writer << "\nauto bwd_desc = "
                              "dnnl::convolution_backward_weights::desc(\n";
                    writer << conv_algo_string;
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 2 << "],\n";
                    if (has_bias)
                    {
                        writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 3 << "],\n";
                    }
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(strides);
                    WRITE_DNNL_DIMS(window_dilation_strides_adjusted);
                    WRITE_DNNL_DIMS(pad_below);
                    writer << "dnnl::memory::dims{" << join(pad_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create forward primitive descriptor\n";
                    writer << "auto fwd_pd = dnnl::convolution_forward::primitive_desc(fwd_desc, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// create backward primitive_descriptor\n";
                    writer << "auto bwd_pd = "
                              "dnnl::convolution_backward_weights::primitive_desc(bwd_desc, "
                              "attr, "
                              "cg_ctx->global_cpu_engine, fwd_pd);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::convolution_backward_weights(bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    ConvolutionBackpropFilters)
                {
                    construct_primitive_build_string_conv_backward_filters<
                        ConvolutionBackpropFilters>(dnnl_emitter,
                                                    node,
                                                    construct_string,
                                                    deps,
                                                    index,
                                                    scratchpad_size,
                                                    desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    ConvolutionBiasBackpropFiltersBias)
                {
                    construct_primitive_build_string_conv_backward_filters<
                        ConvolutionBiasBackpropFiltersBias>(dnnl_emitter,
                                                            node,
                                                            construct_string,
                                                            deps,
                                                            index,
                                                            scratchpad_size,
                                                            desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    ConvolutionBackpropData)
                {
                    auto convolution = static_cast<const ConvolutionBackpropData*>(node);

                    // query scratchpad size
                    auto bwd_desc = dnnl_emitter.get_convolution_backward_data_desc<
                        ngraph::op::v0::ConvolutionBackpropData>(node);
                    auto fwd_desc = dnnl_emitter.get_convolution_forward_desc_for_backward_op<
                        ngraph::op::v0::ConvolutionBackpropData>(node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_convolution_backward_data(fwd_desc, bwd_desc);

                    Strides window_dilation_strides_adjusted;
                    for (size_t s : convolution->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto arg0_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto arg1_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto out0_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    auto strides = convolution->get_window_movement_strides_forward();
                    auto pad_below = convolution->get_padding_below_forward();
                    auto pad_above = convolution->get_padding_above_forward();
                    dnnl::algorithm conv_algo = dnnl_utils::get_conv_algo();
                    auto conv_algo_string = conv_algo == dnnl::algorithm::convolution_auto
                                                ? "dnnl::algorithm::convolution_auto,\n"
                                                : "dnnl::algorithm::convolution_direct,\n";

                    std::vector<dnnl::memory::desc> descs = {arg0_desc, arg1_desc, out0_desc};
                    // ConvolutionBackpropData needs 4 primitives: weights, diff_dst, diff_src,
                    // and convolution_backward_data.
                    index = dnnl_emitter.reserve_primitive_space(4);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto fwd_desc = "
                              "dnnl::convolution_forward::desc(dnnl::prop_kind::forward,\n";
                    writer << conv_algo_string;
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 2
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "],\n";
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(strides);
                    WRITE_DNNL_DIMS(window_dilation_strides_adjusted);
                    WRITE_DNNL_DIMS(pad_below);
                    writer << "dnnl::memory::dims{" << join(pad_above) << "});\n";

                    writer << "\nauto bwd_desc = "
                              "dnnl::convolution_backward_data::desc(\n";
                    writer << conv_algo_string;
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 2
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "],\n";
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(strides);
                    WRITE_DNNL_DIMS(window_dilation_strides_adjusted);
                    WRITE_DNNL_DIMS(pad_below);
                    writer << "dnnl::memory::dims{" << join(pad_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create forward primitive descriptor\n";
                    writer << "auto fwd_pd = dnnl::convolution_forward::primitive_desc(fwd_desc, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// create backward primitive_descriptor\n";
                    writer << "auto bwd_pd = "
                              "dnnl::convolution_backward_data::primitive_desc(bwd_desc, attr, "
                              "cg_ctx->global_cpu_engine, fwd_pd);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::convolution_backward_data(bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void
                    DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(DeconvolutionBias)
                {
                    auto dconv = static_cast<const DeconvolutionBias*>(node);

                    // query scratchpad size
                    auto deconvbias_desc =
                        dnnl_emitter
                            .get_deconvolutionbias_forward_data<ngraph::op::DeconvolutionBias>(
                                node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_deconvolution_forward(deconvbias_desc);

                    // For dilation, DNNL wants to know how many elements to insert between, not
                    // how far
                    // apart to space the elements like nGraph. So we have to subtract 1 from each
                    // pos.
                    Strides window_dilation_strides_adjusted;

                    for (size_t s : dconv->get_window_dilation_strides_forward())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    auto window_strides = dconv->get_window_movement_strides_forward();
                    auto padding_below = dconv->get_padding_below_forward();
                    auto padding_above = dconv->get_padding_above_forward();

                    CodeWriter writer;
                    std::vector<dnnl::memory::desc> descs = {
                        weights_desc, delta_desc, bias_desc, result_desc};

                    // DeconvolutionBias needs 5 primitives: weights, delta, bias, result,
                    // and deconvolutionbias.
                    index = dnnl_emitter.reserve_primitive_space(5);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    if (dconv->with_relu())
                    {
                        writer << "dnnl::post_ops pops;\n";
                        writer << "const float ops_scale = 1.f;\n";
                        writer << "const float ops_alpha = -0.f; // relu negative slope\n";
                        writer << "const float ops_beta = 0.f;\n";

                        writer << "pops.append_eltwise("
                                  "ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, "
                                  "ops_beta);\n";
                    }
                    else
                    {
                        writer << "dnnl::post_ops pops = dnnl::post_ops();\n";
                    }

                    writer << "dnnl::primitive_attr dconv_attr;\n";
                    writer << "dconv_attr.set_post_ops(pops);\n";
                    writer << "dconv_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\nauto dconv_desc = "
                              "dnnl::deconvolution_forward::desc(\n"
                              "dnnl::prop_kind::forward,\n"
                              "dnnl::algorithm::deconvolution_direct,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 0
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 2
                           << "],\n"

                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 3 << "],\n";

                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_dilation_strides_adjusted);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "\n// create forward primitive descriptor\n";
                    writer << "auto dconv_pd = "
                              "dnnl::deconvolution_forward::primitive_desc(dconv_desc, "
                              "dconv_attr, cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::deconvolution_forward(dconv_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(dconv_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <typename OP>
                void construct_primitive_build_string_max_pool(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file)
                {
                    auto pool = static_cast<const OP*>(node);
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto max_pool_desc =
                        dnnl_emitter.get_max_pooling_forward_desc<ngraph::op::v0::MaxPool>(node,
                                                                                           false);
                    scratchpad_size = dnnl_emitter.query_scratchpad_pooling_forward(max_pool_desc);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    CodeWriter writer;
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};

                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "\n// build Maxpool primitive descriptor\n";
                    writer << "auto max_pool_desc = ";
                    writer << "dnnl::pooling_forward::desc(dnnl::prop_kind::forward_"
                              "inference,\n";
                    writer << "dnnl::algorithm::pooling_max,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "auto max_pool_pd = dnnl::pooling_forward::primitive_desc("
                              "max_pool_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::pooling_forward(max_pool_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(max_pool_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <typename OP>
                void construct_primitive_build_string_avg_pool(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file)
                {
                    auto pool = static_cast<const OP*>(node);

                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto avg_pool_desc =
                        dnnl_emitter.get_avg_pooling_forward_desc<ngraph::op::v0::AvgPool>(node,
                                                                                           false);
                    scratchpad_size = dnnl_emitter.query_scratchpad_pooling_forward(avg_pool_desc);

                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();
                    auto include_padding_in_avg_computation =
                        pool->get_include_padding_in_avg_computation();

                    CodeWriter writer;
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};

                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "\n// build Avgpool primitive descriptor\n";
                    writer << "auto avg_pool_desc = ";
                    writer << "dnnl::pooling_forward::desc(dnnl::prop_kind::forward_"
                              "inference,\n";
                    if (include_padding_in_avg_computation)
                    {
                        writer << "dnnl::algorithm::pooling_avg_include_padding,\n";
                    }
                    else
                    {
                        writer << "dnnl::algorithm::pooling_avg_exclude_padding,\n";
                    }
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "auto avg_pool_pd = dnnl::pooling_forward::primitive_desc("
                              "avg_pool_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::pooling_forward(avg_pool_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(avg_pool_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(MaxPool)
                {
                    construct_primitive_build_string_max_pool<MaxPool>(dnnl_emitter,
                                                                       node,
                                                                       construct_string,
                                                                       deps,
                                                                       index,
                                                                       scratchpad_size,
                                                                       desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(AvgPool)
                {
                    construct_primitive_build_string_avg_pool<AvgPool>(dnnl_emitter,
                                                                       node,
                                                                       construct_string,
                                                                       deps,
                                                                       index,
                                                                       scratchpad_size,
                                                                       desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    MaxPoolWithIndices)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    auto pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);
                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    // query scratchpad size
                    auto max_pool_desc = dnnl_emitter.get_max_pooling_with_indices_forward_desc<
                        ngraph::op::MaxPoolWithIndices>(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_pooling_forward(max_pool_desc);

                    // MaxPoolWithIndices needs 4 primitives: input, result, workspace, and
                    // pooling_forward.
                    index = dnnl_emitter.reserve_primitive_space(4);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto pool_desc = "
                              "dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,\n"
                              "dnnl::algorithm::pooling_max,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build primitive descriptor\n";
                    writer << "dnnl::pooling_forward::primitive_desc fwd_pd{pool_desc, "
                              "cg_ctx->global_cpu_engine};\n";
                    writer << "cg_ctx->dnnl_memories[" << std::to_string(deps[2])
                           << "] = new dnnl::memory(fwd_pd.workspace_desc(), "
                              "cg_ctx->global_cpu_engine, nullptr);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::pooling_forward(fwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(fwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(AvgPoolBackprop)
                {
                    auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    auto pool = static_cast<const ngraph::op::v0::AvgPoolBackprop*>(node);
                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();
                    auto algo_string = pool->get_include_padding_in_avg_computation()
                                           ? "dnnl::algorithm::pooling_avg_include_padding"
                                           : "dnnl::algorithm::pooling_avg_exclude_padding";

                    // query scratchpad size
                    auto avg_pool_fwd_desc =
                        dnnl_emitter.get_avg_pooling_forward_desc<ngraph::op::v0::AvgPoolBackprop>(
                            node, true);
                    auto avg_pool_desc =
                        dnnl_emitter.get_avg_pooling_backward_desc<ngraph::op::v0::AvgPoolBackprop>(
                            node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_avg_pooling_backward(
                        avg_pool_fwd_desc, avg_pool_desc);

                    // AvgPoolBackprop needs 3 primitives: diff_dst, diff_src, and pooling_backward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {diff_dst_desc, diff_src_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto fwd_desc = "
                              "dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,\n";
                    writer << algo_string << ",\n";
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 1
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "auto bwd_desc = "
                              "dnnl::pooling_backward::desc(\n";
                    writer << algo_string << ",\n";
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 1
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build primitive descriptor\n";
                    writer << "dnnl::pooling_forward::primitive_desc fwd_pd{fwd_desc, "
                              "cg_ctx->global_cpu_engine};\n";
                    writer << "dnnl::pooling_backward::primitive_desc bwd_pd{bwd_desc, attr, "
                              "cg_ctx->global_cpu_engine, fwd_pd};\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::pooling_backward(bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(MaxPoolBackprop)
                {
                    auto fprop_src_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    auto pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);
                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    // query scratchpad size
                    auto fwd_pool_desc =
                        dnnl_emitter.get_max_pooling_forward_desc<ngraph::op::v0::MaxPoolBackprop>(
                            node, true);
                    auto bwd_pool_desc =
                        dnnl_emitter.get_max_pooling_backward_desc<ngraph::op::v0::MaxPoolBackprop>(
                            node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_max_pooling_backward(
                        fwd_pool_desc, bwd_pool_desc);

                    // MaxPoolBackprop needs 6 primitives: fprop_src, diff_dst, diff_src, workspace
                    // pooling forward, and pooling_backward.
                    // It needs a new workspace.
                    index = dnnl_emitter.reserve_primitive_space(
                        6, true /* fwd and bwd */, true /* new workspace */);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {
                        fprop_src_desc, diff_dst_desc, diff_src_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto fwd_desc = "
                              "dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,\n"
                              "dnnl::algorithm::pooling_max,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 2
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "\nauto bwd_desc = "
                              "dnnl::pooling_backward::desc(\n"
                              "dnnl::algorithm::pooling_max,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 2
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build primitive descriptor\n";
                    writer << "dnnl::pooling_forward::primitive_desc fwd_pd{fwd_desc, attr, "
                              "cg_ctx->global_cpu_engine};\n";
                    writer << "dnnl::pooling_backward::primitive_desc bwd_pd{bwd_desc, attr, "
                              "cg_ctx->global_cpu_engine, fwd_pd};\n";

                    // This is implemented differently from cpu builder,
                    // we only use one index and one deps here.
                    writer << "cg_ctx->dnnl_memories[" << std::to_string(deps[3])
                           << "] = new dnnl::memory(fwd_pd.workspace_desc(), "
                              "cg_ctx->global_cpu_engine, nullptr);\n";
                    writer << "auto workspace = "
                              "(char*)malloc(fwd_pd.workspace_desc().get_size());"
                              "\n";
                    writer << "if (!workspace)\n";
                    writer.block_begin();
                    writer << "throw std::bad_alloc();\n";
                    writer.block_end();
                    writer << "cg_ctx->dnnl_workspaces.push_back(workspace);\n";

                    deps[4] = dnnl_emitter.reserve_workspace();

                    writer << "\n// build primitive\n";

                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index - 1)
                           << "] = new dnnl::pooling_forward(fwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index - 1)
                           << "] = new dnnl::memory::desc(fwd_pd.scratchpad_desc());\n";

                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::pooling_backward(bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    MaxPoolWithIndicesBackprop)
                {
                    auto diff_dst_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto diff_src_desc = dnnl_utils::get_output_dnnl_md(node, 0);
                    auto pool = static_cast<const ngraph::op::MaxPoolWithIndices*>(node);
                    auto window_shape = pool->get_window_shape();
                    auto window_strides = pool->get_window_movement_strides();
                    auto padding_below = pool->get_padding_below();
                    auto padding_above = pool->get_padding_above();

                    // query scratchpad size
                    auto fwd_pool_desc =
                        dnnl_emitter
                            .get_max_pooling_forward_desc<ngraph::op::MaxPoolWithIndicesBackprop>(
                                node, true);
                    auto bwd_pool_desc =
                        dnnl_emitter
                            .get_max_pooling_backward_desc<ngraph::op::MaxPoolWithIndicesBackprop>(
                                node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_max_pooling_with_indices_backward(
                            fwd_pool_desc, bwd_pool_desc);

                    // MaxPoolWithIndicesBackprop needs 4 primitives: diff_dst, fprop_workspace,
                    // diff_src
                    // and pooling_backward.
                    index = dnnl_emitter.reserve_primitive_space(4);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {diff_dst_desc, diff_src_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto fwd_desc = "
                              "dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,\n"
                              "dnnl::algorithm::pooling_max,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "auto bwd_desc = "
                              "dnnl::pooling_backward::desc(\n"
                              "dnnl::algorithm::pooling_max,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "],\n";
                    WRITE_DNNL_DIMS(window_strides);
                    WRITE_DNNL_DIMS(window_shape);
                    WRITE_DNNL_DIMS(padding_below);
                    writer << "dnnl::memory::dims{" << join(padding_above) << "});\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build primitive descriptor\n";
                    writer << "dnnl::pooling_forward::primitive_desc fwd_pd{fwd_desc, "
                              "cg_ctx->global_cpu_engine};\n";
                    writer << "dnnl::pooling_backward::primitive_desc bwd_pd{bwd_desc, attr, "
                              "cg_ctx->global_cpu_engine, fwd_pd};\n";
                    // this is different from cpu builder because we do not write workspace desc to
                    // desc_file.
                    // here workspace's dnnl primitive index is in deps[2] in stead of deps[1].
                    writer << "cg_ctx->dnnl_memories[" << std::to_string(deps[2])
                           << "] = new dnnl::memory(fwd_pd.workspace_desc(), "
                              "cg_ctx->global_cpu_engine, nullptr);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::pooling_backward(bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(
                    ngraph::runtime::cpu::op::ConvertLayout)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    bool input_format_is_nchw = dnnl_utils::dnnl_md_matches_format_tag(
                        input_desc.data, dnnl::memory::format_tag::nchw);
                    if (input_format_is_nchw &&
                        dnnl_utils::dnnl_md_matches_format_tag(result_desc.data,
                                                               dnnl::memory::format_tag::goihw))
                    {
                        // becomes a copy
                        input_desc = result_desc;
                    }
                    else if ((input_format_is_nchw ||
                              dnnl_utils::dnnl_md_matches_format_tag(
                                  input_desc.data, dnnl::memory::format_tag::nhwc)) &&
                             (dnnl_utils::dnnl_md_matches_format_tag(
                                  result_desc.data, dnnl::memory::format_tag::OIhw4i16o4i) &&
                              // check if compensation is conv_s8s8(1U)
                              result_desc.data.extra.flags & 0x1U))
                    {
                        auto arg0_shape = node->get_input_shape(0);
                        input_desc = dnnl::memory::desc(
                            dnnl::memory::dims(arg0_shape.begin(), arg0_shape.end()),
                            dnnl_utils::get_dnnl_data_type(node->get_input_element_type(0)),
                            dnnl::memory::format_tag::oihw);
                    }
                    else if (input_format_is_nchw && input_desc.data.ndims == 4 &&
                             result_desc.data.ndims == 5 && node->get_users().size() == 1)
                    {
                        Shape weights_shape_groups;
                        if (auto gconv =
                                as_type_ptr<ngraph::op::v0::GroupConvolution>(node->get_users()[0]))
                        {
                            weights_shape_groups = gconv->get_weights_dimensions();
                        }
                        else if (auto gconvb = as_type_ptr<ngraph::op::GroupConvolutionBias>(
                                     node->get_users()[0]))
                        {
                            weights_shape_groups = gconvb->get_weights_dimensions();
                        }
                        else
                        {
                            throw ngraph_error(
                                "Incompatible input/output shape in ConvertLayout op");
                        }
                        input_desc = dnnl::memory::desc(
                            dnnl::memory::dims(weights_shape_groups.begin(),
                                               weights_shape_groups.end()),
                            dnnl_utils::get_dnnl_data_type(node->get_input_element_type(0)),
                            dnnl::memory::format_tag::goihw);
                    }

                    // query scratchpad size
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_reorder(input_desc, result_desc);

                    // ConvertLayout needs 3 primitives: input, result, and reorder.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build reorder primitive\n";
                    writer << "auto reorder_pd = "
                              "dnnl::reorder::primitive_desc("
                              "*cg_ctx->dnnl_memories["
                           << std::to_string(deps[0])
                           << "]"
                              ", *cg_ctx->dnnl_memories["
                           << std::to_string(deps[1]) << "], attr);\n";

                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::reorder(reorder_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(reorder_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(ReluBackprop)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto bwd_desc = dnnl_emitter.get_relu_backward_desc(node);
                    auto fwd_desc = dnnl_emitter.get_relu_forward_desc(node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_eltwise_backward(fwd_desc, bwd_desc);

                    // ReluBackprop needs 4 primitives: input, delta, result, and eltwise_backward.
                    index = dnnl_emitter.reserve_primitive_space(4);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, delta_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "const float negative_slope = 0.0f;\n";
                    writer << "auto fwd_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, "
                              "dnnl::algorithm::eltwise_relu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], negative_slope);\n";
                    writer << "auto bwd_desc = "
                              "dnnl::eltwise_backward::desc(dnnl::algorithm::eltwise_relu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 2
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], negative_slope);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create forward relu primitive descriptor\n";
                    writer << "auto relu_fwd_pd = dnnl::eltwise_forward::primitive_desc(fwd_desc, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// create backward relu primitive_descriptor\n";
                    writer << "auto relu_bwd_pd = "
                              "dnnl::eltwise_backward::primitive_desc(bwd_desc, attr, "
                              "cg_ctx->global_cpu_engine, relu_fwd_pd);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_backward(relu_bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(relu_bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Relu)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto relu_desc = dnnl_emitter.get_relu_forward_desc(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_eltwise_forward(relu_desc);

                    // Relu needs 3 primitives: input, result, and eltwise_forward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "const float negative_slope = 0.0f;\n";
                    writer << "auto relu_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, "
                              "dnnl::algorithm::eltwise_relu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], negative_slope);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create relu primitive_descriptor\n";
                    writer << "auto relu_pd = "
                              "dnnl::eltwise_forward::primitive_desc(relu_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_forward(relu_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(relu_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(CPULeakyRelu)
                {
                    auto leaky_relu_node = static_cast<const ngraph::op::CPULeakyRelu*>(node);
                    float alpha = leaky_relu_node->get_alpha();
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto leaky_relu_desc = dnnl_emitter.get_leaky_relu_desc(node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_eltwise_forward(leaky_relu_desc);

                    // CPULeakyRelu needs 3 primitives: input, result, and eltwise_forward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "const float alpha = " << alpha << ";\n";
                    writer << "auto relu_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, "
                              "dnnl::algorithm::eltwise_relu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], alpha, 0.0f);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create relu primitive_descriptor\n";
                    writer << "auto relu_pd = "
                              "dnnl::eltwise_forward::primitive_desc(relu_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_forward(relu_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(relu_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(BoundedRelu)
                {
                    auto bounded_relu_node = static_cast<const ngraph::op::BoundedRelu*>(node);
                    float alpha = bounded_relu_node->get_alpha();
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto bounded_relu_desc = dnnl_emitter.get_bounded_relu_desc(node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_eltwise_forward(bounded_relu_desc);

                    // BoundedRelu needs 3 primitives: input, result, and eltwise_forward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "const float alpha = " << alpha << ";\n";
                    writer << "auto relu_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, "
                              "dnnl::algorithm::eltwise_bounded_relu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], alpha, 0.0f);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create relu primitive_descriptor\n";
                    writer << "auto relu_pd = "
                              "dnnl::eltwise_forward::primitive_desc(relu_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_forward(relu_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(relu_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Gelu)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto gelu_desc = dnnl_emitter.get_gelu_forward_desc(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_eltwise_forward(gelu_desc);

                    // Gelu needs 3 primitives: input, result, and eltwise_forward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto gelu_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, "
                              "dnnl::algorithm::eltwise_gelu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], 1.0f, 0.0f);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create gelu primitive_descriptor\n";
                    writer << "auto gelu_pd = "
                              "dnnl::eltwise_forward::primitive_desc(gelu_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_forward(gelu_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(gelu_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }
                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(GeluBackprop)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto fwd_desc = dnnl_emitter.get_gelu_forward_desc(node);
                    auto bwd_desc = dnnl_emitter.get_gelu_backward_desc(node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_eltwise_backward(fwd_desc, bwd_desc);

                    // GeluBackprop needs 4 primitives: input, delta, result, and
                    // eltwise_backward.
                    index = dnnl_emitter.reserve_primitive_space(4);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, delta_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto fwd_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, "
                              "dnnl::algorithm::eltwise_gelu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], 0, 0);\n";
                    writer << "auto bwd_desc = "
                              "dnnl::eltwise_backward::desc(dnnl::algorithm::eltwise_gelu, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], 0, 0);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create forward gelu primitive descriptor\n";
                    writer << "auto gelu_fwd_pd = "
                              "dnnl::eltwise_forward::primitive_desc(fwd_desc, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// create backward gelu primitive_descriptor\n";
                    writer << "auto gelu_bwd_pd = "
                              "dnnl::eltwise_backward::primitive_desc(bwd_desc, attr, "
                              "cg_ctx->global_cpu_engine, gelu_fwd_pd);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_backward(gelu_bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(gelu_bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Sigmoid)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto sigmoid_desc = dnnl_emitter.get_sigmoid_forward_desc(node, false);
                    scratchpad_size = dnnl_emitter.query_scratchpad_eltwise_forward(sigmoid_desc);

                    // Sigmoid needs 3 primitives: input, result, and eltwise_forward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto sigmoid_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, "
                              "dnnl::algorithm::eltwise_logistic, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], 0, 0);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create sigmoid primitive_descriptor\n";
                    writer << "auto sigmoid_pd = "
                              "dnnl::eltwise_forward::primitive_desc(sigmoid_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_forward(sigmoid_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(sigmoid_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(SigmoidBackprop)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto delta_desc = dnnl_utils::get_input_dnnl_md(node, 1);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto fwd_desc = dnnl_emitter.get_sigmoid_forward_desc(node, true);
                    auto bwd_desc = dnnl_emitter.get_sigmoid_backward_desc(node);
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_eltwise_backward(fwd_desc, bwd_desc);

                    // SigmoidBackprop needs 4 primitives: input, delta, result, and
                    // eltwise_backward.
                    index = dnnl_emitter.reserve_primitive_space(4);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, delta_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto fwd_desc = "
                              "dnnl::eltwise_forward::desc(dnnl::prop_kind::forward, "
                              "dnnl::algorithm::eltwise_logistic, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], 0, 0);\n";
                    writer << "auto bwd_desc = "
                              "dnnl::eltwise_backward::desc(dnnl::algorithm::eltwise_logistic, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1
                           << "], "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], 0, 0);\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create forward sigmoid primitive descriptor\n";
                    writer << "auto sigmoid_fwd_pd = "
                              "dnnl::eltwise_forward::primitive_desc(fwd_desc, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// create backward sigmoid primitive_descriptor\n";
                    writer << "auto sigmoid_bwd_pd = "
                              "dnnl::eltwise_backward::primitive_desc(bwd_desc, attr, "
                              "cg_ctx->global_cpu_engine, sigmoid_fwd_pd);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::eltwise_backward(sigmoid_bwd_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(sigmoid_bwd_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Softmax)
                {
                    auto softmax = static_cast<const ngraph::op::v0::Softmax*>(node);

                    if (softmax->get_axes().size() != 1)
                    {
                        throw ngraph_error("DNNL supports softmax only across single axis");
                    }

                    int softmax_axis = static_cast<int>(*(softmax->get_axes().begin()));
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto softmax_desc = dnnl_emitter.get_softmax_forward_desc(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_softmax_forward(softmax_desc);

                    // Softmax needs 3 primitives: input, result, and softmax_forward.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "auto softmax_desc = "
                              "dnnl::softmax_forward::desc(dnnl::prop_kind::forward_scoring, "
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index << "], " << softmax_axis << ");\n";

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// create softmax primitive_descriptor\n";
                    writer << "auto softmax_pd = "
                              "dnnl::softmax_forward::primitive_desc(softmax_desc, attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::softmax_forward(softmax_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(softmax_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Quantize)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_reorder(input_desc, result_desc);

                    // Quantize needs 3 primitives: input, result, and reorder.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_output_scales(mask, dyn_scales);\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build reorder primitive\n";
                    writer << "auto reorder_pd = "
                              "dnnl::reorder::primitive_desc("
                              "*cg_ctx->dnnl_memories["
                           << std::to_string(deps[0])
                           << "]"
                              ", *cg_ctx->dnnl_memories["
                           << std::to_string(deps[1]) << "], attr);\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::reorder(reorder_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(reorder_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Dequantize)
                {
                    auto input_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    scratchpad_size =
                        dnnl_emitter.query_scratchpad_reorder(input_desc, result_desc);

                    // Dequantize needs 3 primitives: input, result, and reorder.
                    index = dnnl_emitter.reserve_primitive_space(3);
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {input_desc, result_desc};
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "dnnl::primitive_attr attr;\n";
                    writer << "attr.set_output_scales(mask, dyn_scales);\n";
                    writer << "attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    writer << "\n// build reorder primitive\n";
                    writer << "auto reorder_pd = "
                              "dnnl::reorder::primitive_desc("
                              "*cg_ctx->dnnl_memories["
                           << std::to_string(deps[0])
                           << "]"
                              ", *cg_ctx->dnnl_memories["
                           << std::to_string(deps[1]) << "], attr);\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::reorder(reorder_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(reorder_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <typename OP>
                void construct_primitive_build_string_inner_product(
                    ngraph::runtime::cpu::DNNLEmitter& dnnl_emitter,
                    ngraph::Node* node,
                    std::string& construct_string,
                    std::vector<size_t>& deps,
                    size_t& index,
                    size_t& scratchpad_size,
                    std::ofstream& desc_file)
                {
                    auto has_bias = false;
                    if (dnnl_emitter.has_bias<OP>())
                    {
                        has_bias = true;
                    }

                    auto data_desc = dnnl_utils::get_input_dnnl_md(node, 0);
                    auto weights_desc = dnnl_utils::get_input_dnnl_md(node, 1);

                    auto result_desc = dnnl_utils::get_output_dnnl_md(node, 0);

                    // query scratchpad size
                    auto ip_desc = dnnl_emitter.get_inner_product_forward_desc<OP>(node);
                    auto ip_attr = dnnl_emitter.get_inner_product_forward_attr<OP>(node);
                    scratchpad_size = dnnl_emitter.query_scratchpad_ip_forward(ip_desc, ip_attr);

                    if (has_bias)
                    {
                        // QuantizedDotBias needs 5 primitives: input, weights, bias, result, and
                        // inner_product.
                        index = dnnl_emitter.reserve_primitive_space(5);
                    }
                    else
                    {
                        // QuantizedDot needs 4 primitives: input, weights, result, and
                        // inner_product.
                        index = dnnl_emitter.reserve_primitive_space(4);
                    }
                    deps = dnnl_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    // Write memory descriptors to file
                    std::vector<dnnl::memory::desc> descs = {data_desc, weights_desc, result_desc};

                    if (has_bias)
                    {
                        auto bias_desc = dnnl_utils::get_input_dnnl_md(node, 2);
                        descs.push_back(bias_desc);
                    }

                    auto desc_index = dnnl_emitter.get_dnnl_descriptors_size();
                    dnnl_emitter.reserve_descriptor_space(descs.size());
                    serialize_memory_descs(desc_file, descs, deps[0]);

                    writer << "\n// build primitive descriptor\n";
                    writer << "auto ip_desc = "
                              "dnnl::inner_product_forward::desc(dnnl::prop_kind::forward,\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index
                           << "],\n"
                              "*cg_ctx->dnnl_descriptors["
                           << desc_index + 1 << "],\n";
                    if (has_bias)
                    {
                        writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 3 << "],\n";
                    }
                    writer << "*cg_ctx->dnnl_descriptors[" << desc_index + 2 << "]);\n";

                    writer << "\ndnnl::post_ops ops;\n";
                    if (std::is_same<OP, ngraph::op::v0::QuantizedDotBias>() &&
                        has_relu<ngraph::op::v0::QuantizedDotBias>(node))
                    {
                        writer << "const float ops_scale = 1.f;\n";
                        writer << "const float ops_alpha = -0.f; // relu negative slope\n";
                        writer << "const float ops_beta = 0.f;\n";
                        writer << "ops.append_eltwise("
                                  "ops_scale, dnnl::algorithm::eltwise_relu, ops_alpha, "
                                  "ops_beta);\n";
                    }

                    writer << "dnnl::primitive_attr ip_attr;\n";
                    writer << "ip_attr.set_post_ops(ops);\n";
                    writer << "ip_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);\n";

                    if (dnnl_emitter.is_quantized_inner_product<OP>())
                    {
                        writer << "ip_attr.set_output_scales(mask, dyn_scales);\n";
                    }

                    writer << "auto ip_pd = "
                              "dnnl::inner_product_forward::primitive_desc(ip_desc, ip_attr, "
                              "cg_ctx->global_cpu_engine);\n";

                    writer << "\n// build primitive\n";
                    writer << "cg_ctx->dnnl_primitives[" << std::to_string(index)
                           << "] = new dnnl::inner_product_forward(ip_pd);\n";
                    writer << "cg_ctx->dnnl_scratchpad_mds[" << std::to_string(index)
                           << "] = new dnnl::memory::desc(ip_pd.scratchpad_desc());\n";

                    construct_string = writer.get_code();
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(QuantizedDotBias)
                {
                    construct_primitive_build_string_inner_product<QuantizedDotBias>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file);
                }

                template <>
                void DNNLPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(QuantizedMatmul)
                {
                    construct_primitive_build_string_inner_product<QuantizedMatmul>(
                        dnnl_emitter,
                        node,
                        construct_string,
                        deps,
                        index,
                        scratchpad_size,
                        desc_file);
                }
            }
        }
    }
}

using namespace ngraph::runtime::cpu::pass;

#define TI(x) std::type_index(typeid(x))

static const PrimitiveBuildStringConstructOpMap prim_build_string_construct_dispatcher{
    {TI(Add), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Add>},
    {TI(BoundedRelu), &DNNLPrimitiveBuildPass::construct_primitive_build_string<BoundedRelu>},
    {TI(Concat), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Concat>},
    {TI(runtime::cpu::op::ConvertLayout),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<runtime::cpu::op::ConvertLayout>},
    {TI(BatchNormInference),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<BatchNormInference>},
    {TI(BatchNormTraining),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<BatchNormTraining>},
    {TI(BatchNormInferenceRelu),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<BatchNormInferenceRelu>},
    {TI(BatchNormTrainingRelu),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<BatchNormTrainingRelu>},
    {TI(BatchNormTrainingBackprop),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<BatchNormTrainingBackprop>},
    {TI(CPULeakyRelu), &DNNLPrimitiveBuildPass::construct_primitive_build_string<CPULeakyRelu>},
    {TI(LRN), &DNNLPrimitiveBuildPass::construct_primitive_build_string<LRN>},
    {TI(Lstm), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Lstm>},
    {TI(Relu), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Relu>},
    {TI(ReluBackprop), &DNNLPrimitiveBuildPass::construct_primitive_build_string<ReluBackprop>},
    {TI(Rnn), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Rnn>},
    {TI(Convolution), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Convolution>},
    {TI(ConvolutionRelu),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<ConvolutionRelu>},
    {TI(ConvolutionBias),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<ConvolutionBias>},
    {TI(ConvolutionBiasAdd),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<ConvolutionBiasAdd>},
    {TI(ConvolutionAdd), &DNNLPrimitiveBuildPass::construct_primitive_build_string<ConvolutionAdd>},
    {TI(GroupConvolution),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<GroupConvolution>},
    {TI(GroupConvolutionBias),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<GroupConvolutionBias>},
    {TI(QuantizedConvolution),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolution>},
    {TI(QuantizedConvolutionRelu),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolutionRelu>},
    {TI(QuantizedConvolutionBias),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolutionBias>},
    {TI(QuantizedConvolutionBiasAdd),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolutionBiasAdd>},
    {TI(QuantizedConvolutionBiasSignedAdd),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<QuantizedConvolutionBiasSignedAdd>},
    {TI(ConvolutionBackpropData),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<ConvolutionBackpropData>},
    {TI(ConvolutionBackpropFilters),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<ConvolutionBackpropFilters>},
    {TI(ConvolutionBiasBackpropFiltersBias),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<ConvolutionBiasBackpropFiltersBias>},
    {TI(DeconvolutionBias),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<DeconvolutionBias>},
    {TI(MaxPoolWithIndices),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<MaxPoolWithIndices>},
    {TI(MaxPoolWithIndicesBackprop),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<MaxPoolWithIndicesBackprop>},
    {TI(Sigmoid), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Sigmoid>},
    {TI(SigmoidBackprop),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<SigmoidBackprop>},
    {TI(Slice), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Slice>},
    {TI(Softmax), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Softmax>},
    {TI(MaxPool), &DNNLPrimitiveBuildPass::construct_primitive_build_string<MaxPool>},
    {TI(AvgPool), &DNNLPrimitiveBuildPass::construct_primitive_build_string<AvgPool>},
    {TI(AvgPoolBackprop),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<AvgPoolBackprop>},
    {TI(MaxPoolBackprop),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<MaxPoolBackprop>},
    {TI(Quantize), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Quantize>},
    {TI(Dequantize), &DNNLPrimitiveBuildPass::construct_primitive_build_string<Dequantize>},
    {TI(QuantizedDotBias),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<QuantizedDotBias>},
    {TI(QuantizedMatmul),
     &DNNLPrimitiveBuildPass::construct_primitive_build_string<QuantizedMatmul>},
};

bool DNNLPrimitiveBuildPass::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    std::ofstream desc_file(m_desc_filename, std::ios::out | std::ios::binary);
    for (const auto& shp_node : nodes)
    {
        Node* node = shp_node.get();

        if (dnnl_utils::use_dnnl_kernel(node))
        {
            auto handler = prim_build_string_construct_dispatcher.find(TI(*node));
            NGRAPH_CHECK(handler != prim_build_string_construct_dispatcher.end(),
                         "Unsupported node '",
                         node->description(),
                         "' in DNNLPrimitiveBuildPass");

            std::string construct_string;
            std::vector<size_t> deps;
            size_t index;
            size_t scratchpad_size;
            handler->second(
                m_dnnl_emitter, node, construct_string, deps, index, scratchpad_size, desc_file);
            m_node_primitive_string_deps_index_size_map[node] =
                std::tuple<std::string, std::vector<size_t>, size_t, size_t>(
                    construct_string, deps, index, scratchpad_size);
        }
    }
    return false;
}

#undef TI
