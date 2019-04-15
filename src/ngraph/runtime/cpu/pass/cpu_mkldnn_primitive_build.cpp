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
                // serialize memory descriptor and emit the code to deserialize memeory descriptors
                static void
                    serialize_and_deserialize_memory_descs(std::ofstream& desc_file,
                                                           CodeWriter& writer,
                                                           std::vector<mkldnn::memory::desc>& descs,
                                                           std::vector<std::string>& desc_names)
                {
                    NGRAPH_ASSERT(descs.size() == desc_names.size());
                    for (auto i = 0; i < descs.size(); i++)
                    {
                        desc_file.write(reinterpret_cast<char*>(&descs[i]),
                                        sizeof(mkldnn::memory::desc));
                        writer << "char " << desc_names[i] << "[sizeof(mkldnn::memory::desc)];\n";
                        writer << "desc_file.read(" << desc_names[i]
                               << ", sizeof(mkldnn::memory::desc));\n";
                    }
                }

                // emit the code to build memory primitives
                static void emit_memory_primitive_build(CodeWriter& writer,
                                                        std::vector<std::string>& desc_names,
                                                        std::vector<size_t>& deps,
                                                        bool new_workspace = false)
                {
                    NGRAPH_ASSERT(desc_names.size() == new_workspace ? deps.size()
                                                                     : deps.size() - 1);
                    for (auto i = 0; i < desc_names.size(); i++)
                    {
                        writer << "cg_ctx->mkldnn_primitives[" << std::to_string(deps[i])
                               << "] = new "
                                  "mkldnn::memory({*reinterpret_cast<mkldnn::memory::desc*>("
                               << desc_names[i] << "), cg_ctx->global_cpu_engine}, nullptr);\n";
                    }
                }

                // The following functions build the MKLDNN primitive for each type of nGraph Node.

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Add)
                {
                    std::vector<float> scale_vector(2, 1);
                    std::vector<mkldnn::memory::primitive_desc> inputs_pd;

                    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input0_data_desc, executor::global_cpu_engine));
                    inputs_pd.push_back(mkldnn::memory::primitive_desc(
                        input1_data_desc, executor::global_cpu_engine));

                    return mkldnn_emitter.build_elementwise_add(
                        input0_data_desc, input1_data_desc, result_desc, scale_vector, inputs_pd);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Add)
                {
                    auto input0_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto input1_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    // Add needs 4 primitives: input0, input1, result, and sum.
                    index = mkldnn_emitter.reserve_primitive_space(4);
                    deps = mkldnn_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    writer << "// read in memory descriptors\n";
                    std::vector<mkldnn::memory::desc> descs = {
                        input0_data_desc, input1_data_desc, result_desc};
                    std::vector<std::string> desc_names = {
                        "input0_data_desc", "input1_data_desc", "result_desc"};
                    serialize_and_deserialize_memory_descs(desc_file, writer, descs, desc_names);

                    writer << "\n// build sum primitive descriptor\n";
                    writer << "std::vector<float> scale_vector(2, 1);\n";
                    writer << "std::vector<mkldnn::memory::primitive_desc> inputs_pd;\n";
                    writer << "inputs_pd.push_back(mkldnn::memory::primitive_desc(*reinterpret_"
                              "cast<mkldnn::memory::desc*>(input0_data_desc), "
                              "cg_ctx->global_cpu_engine));\n";
                    writer << "inputs_pd.push_back(mkldnn::memory::primitive_desc(*reinterpret_"
                              "cast<mkldnn::memory::desc*>(input1_data_desc), "
                              "cg_ctx->global_cpu_engine));\n";

                    // elementwise sum primitive descriptor
                    writer << "mkldnn::sum::primitive_desc sum_pd = "
                              "mkldnn::sum::primitive_desc(*reinterpret_cast<mkldnn::memory::desc*>"
                              "(result_desc), scale_vector, inputs_pd);\n";

                    writer << "\n// build sum primitive\n";
                    writer << "std::vector<mkldnn::memory::primitive::at> inputs_primitive;\n";
                    emit_memory_primitive_build(writer, desc_names, deps);
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
                        case rnn_utils::rnntype::vanilla_rnn: return mkldnn::algorithm::vanilla_rnn;
                        case rnn_utils::rnntype::vanilla_gru: return mkldnn::algorithm::vanilla_gru;
                        case rnn_utils::rnntype::vanilla_lstm:
                            return mkldnn::algorithm::vanilla_lstm;
                        default: throw ngraph_error("unsupported mkldnn rnn algorithm");
                        }
                    };

                    auto get_mkldnn_rnn_direction = [&]() {
                        switch (direction)
                        {
                        case 1: return mkldnn::rnn_direction::unidirectional_left2right;
                        case 2: return mkldnn::rnn_direction::bidirectional_concat;
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
                    // dst_layer, dst_iter, and rnn_forward.
                    // It needs a new workspace.
                    index = mkldnn_emitter.reserve_primitive_space(9, true /* new workspace */);
                    deps = mkldnn_emitter.get_primitive_deps(index);

                    CodeWriter writer;

                    writer << "// read in memory descriptors\n";
                    std::vector<mkldnn::memory::desc> descs = {src_layer_md,
                                                               src_iter_md,
                                                               wei_layer_md,
                                                               wei_iter_md,
                                                               bias_md,
                                                               dst_layer_md,
                                                               dst_iter_md};
                    std::vector<std::string> desc_names = {"src_layer_desc",
                                                           "src_iter_desc",
                                                           "weights_layer_desc",
                                                           "weights_iter_desc",
                                                           "bias_desc",
                                                           "dst_layer_desc",
                                                           "dst_iter_desc"};
                    serialize_and_deserialize_memory_descs(desc_file, writer, descs, desc_names);

                    mkldnn::rnn_cell::desc rnn_cell_desc(get_mkldnn_rnn_cell_type());
                    desc_file.write(reinterpret_cast<char*>(&rnn_cell_desc),
                                    sizeof(mkldnn::rnn_cell::desc));
                    writer << "char rnn_cell_desc[sizeof(mkldnn::rnn_cell::desc)];\n";
                    writer << "desc_file.read(rnn_cell_desc, sizeof(mkldnn::rnn_cell::desc));\n";

                    auto rnn_direction = get_mkldnn_rnn_direction();
                    desc_file.write(reinterpret_cast<char*>(&rnn_direction),
                                    sizeof(mkldnn::rnn_direction));
                    writer << "char rnn_direction[sizeof(mkldnn::rnn_direction)];\n";
                    writer << "desc_file.read(rnn_direction, sizeof(mkldnn::rnn_direction));\n";

                    writer << "\n// build lstm/rnn primitive descriptor\n";
                    writer << "auto rnn_desc = "
                              "mkldnn::rnn_forward::desc(mkldnn::prop_kind::forward_training, "
                              "*reinterpret_cast<mkldnn::rnn_cell::desc*>(rnn_cell_desc), "
                              "*reinterpret_cast<mkldnn::rnn_direction*>(rnn_direction), "
                              "*reinterpret_cast<mkldnn::memory::desc*>(src_layer_desc), "
                              "*reinterpret_cast<mkldnn::memory::desc*>(src_iter_desc), "
                              "*reinterpret_cast<mkldnn::memory::desc*>(weights_layer_desc),"
                              "*reinterpret_cast<mkldnn::memory::desc*>(weights_iter_desc), "
                              "*reinterpret_cast<mkldnn::memory::desc*>(bias_desc), "
                              "*reinterpret_cast<mkldnn::memory::desc*>(dst_layer_desc), "
                              "*reinterpret_cast<mkldnn::memory::desc*>(dst_iter_desc));\n";
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
                    emit_memory_primitive_build(writer, desc_names, deps, true);

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
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Lstm)
                {
                    return mkldnn_emitter.build_rnn<Lstm>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Lstm)
                {
                    construct_primitive_build_string_rnn<Lstm>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Rnn)
                {
                    return mkldnn_emitter.build_rnn<Rnn>(node);
                }

                template <>
                void MKLDNNPrimitiveBuildPass::CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(Rnn)
                {
                    construct_primitive_build_string_rnn<Rnn>(
                        mkldnn_emitter, node, construct_string, deps, index, desc_file);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTraining)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormInference>(
                        node, false /*Append relu*/, true /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormInference)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormInference>(
                        node, false /*Append relu*/, false /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTrainingRelu)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormTrainingRelu>(
                        node, true /*Append relu*/, true /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormInferenceRelu)
                {
                    return mkldnn_emitter.build_batch_norm_primitive<BatchNormInferenceRelu>(
                        node, true /*Append relu*/, false /*Training*/);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(BatchNormTrainingBackprop)
                {
                    const auto& args = node->get_inputs();
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

                    const auto* batchnorm = static_cast<const BatchNormTrainingBackprop*>(node);
                    return mkldnn_emitter.build_batchnorm_backward(weights_desc,
                                                                   input_desc,
                                                                   mean_desc,
                                                                   variance_desc,
                                                                   delta_desc,
                                                                   dinput_desc,
                                                                   dweights_desc,
                                                                   batchnorm->get_eps_value());
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Concat)
                {
                    std::vector<mkldnn::memory::desc> inputs_data_desc;
                    for (size_t i = 0, end = node->get_inputs().size(); i < end; i++)
                    {
                        inputs_data_desc.push_back(mkldnn_utils::get_input_mkldnn_md(node, i));
                    }

                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    size_t concat_dim =
                        (static_cast<const Concat*>(node))->get_concatenation_axis();
                    return mkldnn_emitter.build_concat(inputs_data_desc, result_desc, concat_dim);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(LRN)
                {
                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    const auto* lrn = static_cast<const LRN*>(node);

                    return mkldnn_emitter.build_lrn_forward(input_data_desc,
                                                            result_desc,
                                                            static_cast<float>(lrn->get_alpha()),
                                                            static_cast<float>(lrn->get_beta()),
                                                            static_cast<float>(lrn->get_bias()),
                                                            static_cast<int>(lrn->get_nsize()));
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

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionRelu)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionRelu>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionRelu)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionRelu>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolution)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolution>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(GroupConvolution)
                {
                    Strides window_dilation_strides_adjusted;
                    auto convolution = static_cast<const ngraph::op::GroupConvolution*>(node);
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    return mkldnn_emitter.build_convolution_forward(
                        input_data_desc,
                        weights_desc,
                        result_desc,
                        filter_strides,
                        window_dilation_strides_adjusted,
                        padding_below,
                        padding_above);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(GroupConvolutionBias)
                {
                    Strides window_dilation_strides_adjusted;
                    auto convolution = static_cast<const ngraph::op::GroupConvolutionBias*>(node);
                    for (size_t s : convolution->get_window_dilation_strides())
                    {
                        window_dilation_strides_adjusted.push_back(s - 1);
                    }

                    auto input_data_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                    auto weights_desc = mkldnn_utils::get_input_mkldnn_md(node, 1);
                    auto bias_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto padding_below = convolution->get_padding_below();
                    auto padding_above = convolution->get_padding_above();
                    auto filter_strides = convolution->get_window_movement_strides();

                    const float ops_scale = 1.f;
                    const float ops_alpha = -0.f; // relu negative slope
                    const float ops_beta = 0.f;

                    mkldnn::post_ops ops;
                    if (convolution->with_relu())
                    {
                        ops.append_eltwise(
                            ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                    }

                    return mkldnn_emitter.build_convolution_forward(
                        input_data_desc,
                        weights_desc,
                        bias_desc,
                        result_desc,
                        filter_strides,
                        window_dilation_strides_adjusted,
                        padding_below,
                        padding_above,
                        ops);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(Convolution)
                {
                    return mkldnn_emitter.build_convolution<Convolution>(node);
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
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionBias)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionBias>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(QuantizedConvolutionBiasAdd)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionBiasAdd>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(
                    QuantizedConvolutionBiasSignedAdd)
                {
                    return mkldnn_emitter.build_convolution<QuantizedConvolutionBiasSignedAdd>(
                        node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBias)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionBias>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionBiasAdd)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionBiasAdd>(node);
                }

                template <>
                size_t MKLDNNPrimitiveBuildPass::BUILD_PRIMITIVE_DECL(ConvolutionAdd)
                {
                    return mkldnn_emitter.build_convolution<ConvolutionAdd>(node);
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
    {TI(Add), &MKLDNNPrimitiveBuildPass::build_primitive<Add>},
    {TI(Concat), &MKLDNNPrimitiveBuildPass::build_primitive<Concat>},
    {TI(Convert), &MKLDNNPrimitiveBuildPass::build_primitive<Convert>},
    {TI(runtime::cpu::op::ConvertLayout),
     &MKLDNNPrimitiveBuildPass::build_primitive<runtime::cpu::op::ConvertLayout>},
    {TI(AvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPool>},
    {TI(AvgPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<AvgPoolBackprop>},
    {TI(BatchNormTraining), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTraining>},
    {TI(BatchNormInference), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormInference>},
    {TI(BoundedRelu), &MKLDNNPrimitiveBuildPass::build_primitive<BoundedRelu>},
    {TI(BatchNormTrainingBackprop),
     &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTrainingBackprop>},
    {TI(Convolution), &MKLDNNPrimitiveBuildPass::build_primitive<Convolution>},
    {TI(GroupConvolution), &MKLDNNPrimitiveBuildPass::build_primitive<GroupConvolution>},
    {TI(ConvolutionRelu), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionRelu>},
    {TI(ConvolutionBiasAdd), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBiasAdd>},
    {TI(BatchNormTrainingRelu), &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormTrainingRelu>},
    {TI(BatchNormInferenceRelu),
     &MKLDNNPrimitiveBuildPass::build_primitive<BatchNormInferenceRelu>},
    {TI(ConvolutionBackpropData),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropData>},
    {TI(ConvolutionBackpropFilters),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBackpropFilters>},
    {TI(MaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPool>},
    {TI(MaxPoolWithIndices), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndices>},
    {TI(MaxPoolBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolBackprop>},
    {TI(MaxPoolWithIndicesBackprop),
     &MKLDNNPrimitiveBuildPass::build_primitive<MaxPoolWithIndicesBackprop>},
    {TI(ConvolutionBias), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBias>},
    {TI(QuantizedConvolution), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolution>},
    {TI(ConvolutionBiasBackpropFiltersBias),
     &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionBiasBackpropFiltersBias>},
    {TI(LRN), &MKLDNNPrimitiveBuildPass::build_primitive<LRN>},
    {TI(Relu), &MKLDNNPrimitiveBuildPass::build_primitive<Relu>},
    {TI(ReluBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<ReluBackprop>},
    {TI(LeakyRelu), &MKLDNNPrimitiveBuildPass::build_primitive<LeakyRelu>},
    {TI(Sigmoid), &MKLDNNPrimitiveBuildPass::build_primitive<Sigmoid>},
    {TI(SigmoidBackprop), &MKLDNNPrimitiveBuildPass::build_primitive<SigmoidBackprop>},
    {TI(Lstm), &MKLDNNPrimitiveBuildPass::build_primitive<Lstm>},
    {TI(Rnn), &MKLDNNPrimitiveBuildPass::build_primitive<Rnn>},
    {TI(QuantizedMaxPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedMaxPool>},
    {TI(QuantizedAvgPool), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedAvgPool>},
    {TI(Softmax), &MKLDNNPrimitiveBuildPass::build_primitive<Softmax>},
    {TI(Slice), &MKLDNNPrimitiveBuildPass::build_primitive<Slice>},
    {TI(ReplaceSlice), &MKLDNNPrimitiveBuildPass::build_primitive<ReplaceSlice>},
    {TI(UpdateSlice), &MKLDNNPrimitiveBuildPass::build_primitive<UpdateSlice>},
    {TI(ConvolutionAdd), &MKLDNNPrimitiveBuildPass::build_primitive<ConvolutionAdd>},
    {TI(QuantizedConvolutionRelu),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionRelu>},
    {TI(QuantizedConvolutionBias),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBias>},
    {TI(QuantizedConvolutionBiasAdd),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBiasAdd>},
    {TI(QuantizedConvolutionBiasSignedAdd),
     &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConvolutionBiasSignedAdd>},
    {TI(GroupConvolutionBias), &MKLDNNPrimitiveBuildPass::build_primitive<GroupConvolutionBias>},
    {TI(Quantize), &MKLDNNPrimitiveBuildPass::build_primitive<Quantize>},
    {TI(Dequantize), &MKLDNNPrimitiveBuildPass::build_primitive<Dequantize>},
    {TI(QuantizedConcat), &MKLDNNPrimitiveBuildPass::build_primitive<QuantizedConcat>},
    {TI(GetOutputElement), &MKLDNNPrimitiveBuildPass::build_primitive<GetOutputElement>},
};

static const PrimitiveBuildStringConstructOpMap prim_build_string_construct_dispatcher{
    {TI(Add), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Add>},
    {TI(Lstm), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Lstm>},
    {TI(Rnn), &MKLDNNPrimitiveBuildPass::construct_primitive_build_string<Rnn>},
};

bool MKLDNNPrimitiveBuildPass::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
#if 0
    for (const auto& shp_node : nodes)
    {
        Node* node = shp_node.get();

        if (mkldnn_utils::use_mkldnn_kernel(node))
        {
            auto handler = prim_build_dispatcher.find(TI(*node));
            NGRAPH_ASSERT(handler != prim_build_dispatcher.end())
                << "Unsupported node '" << node->description() << "' in MKLDNNPrimitiveBuildPass";

            size_t primitive_idx = handler->second(m_mkldnn_emitter, node);
            m_node_primitive_idx_map[node] = primitive_idx;
        }
    }
#endif

    std::ofstream desc_file = std::ofstream(m_desc_filename, std::ios::out | std::ios::binary);
    for (const auto& shp_node : nodes)
    {
        Node* node = shp_node.get();

        if (mkldnn_utils::use_mkldnn_kernel(node))
        {
            auto handler = prim_build_string_construct_dispatcher.find(TI(*node));
            NGRAPH_ASSERT(handler != prim_build_string_construct_dispatcher.end())
                << "Unsupported node '" << node->description() << "' in MKLDNNPrimitiveBuildPass";

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
