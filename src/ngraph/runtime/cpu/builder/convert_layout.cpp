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

#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::runtime::cpu::op::ConvertLayout)
            {
                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

#if MKLDNN_VERSION_MAJOR < 1
                if (input_desc.data.format == mkldnn_nchw &&
                    result_desc.data.format == mkldnn_goihw)
                {
                    // becomes a copy
                    input_desc = result_desc;
                }
                else if ((input_desc.data.format == mkldnn_nchw ||
                          input_desc.data.format == mkldnn_nhwc) &&
                         result_desc.data.format == mkldnn_OIhw4i16o4i_s8s8)
                {
                    input_desc.data.format = mkldnn_oihw;
                }
                else if (input_desc.data.format == mkldnn_nchw && input_desc.data.ndims == 4 &&
                         result_desc.data.ndims == 5 && node->get_users().size() == 1)
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
                        throw ngraph_error("Incompatible input/output shape in ConvertLayout op");
                    }
                    input_desc = mkldnn::memory::desc(
                        mkldnn::memory::dims(weights_shape_groups.begin(),
                                             weights_shape_groups.end()),
                        mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                        mkldnn::memory::format::goihw);
                }
#else
                bool input_format_is_nchw = mkldnn_utils::mkldnn_md_matches_format_tag(
                    input_desc.data, mkldnn::memory::format_tag::nchw);
                if (input_format_is_nchw &&
                    mkldnn_utils::mkldnn_md_matches_format_tag(result_desc.data,
                                                               mkldnn::memory::format_tag::goihw))
                {
                    // becomes a copy
                    input_desc = result_desc;
                }
                else if ((input_format_is_nchw ||
                          mkldnn_utils::mkldnn_md_matches_format_tag(
                              input_desc.data, mkldnn::memory::format_tag::nhwc)) &&
                         (mkldnn_utils::mkldnn_md_matches_format_tag(
                              result_desc.data, mkldnn::memory::format_tag::OIhw4i16o4i) &&
                          // check if compensation is conv_s8s8(1U)
                          result_desc.data.extra.flags & 0x1U))
                {
                    auto arg0_shape = args[0].get_shape();
                    input_desc = mkldnn::memory::desc(
                        mkldnn::memory::dims(arg0_shape.begin(), arg0_shape.end()),
                        mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                        mkldnn::memory::format_tag::oihw);
                }
                else if (input_format_is_nchw && input_desc.data.ndims == 4 &&
                         result_desc.data.ndims == 5 && node->get_users().size() == 1)
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
                        throw ngraph_error("Incompatible input/output shape in ConvertLayout op");
                    }
                    input_desc = mkldnn::memory::desc(
                        mkldnn::memory::dims(weights_shape_groups.begin(),
                                             weights_shape_groups.end()),
                        mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                        mkldnn::memory::format_tag::goihw);
                }

                mkldnn_emitter->query_scratchpad_reorder(input_desc, result_desc);
#endif
                // ConvertLayout needs 3 primitives: input, result, and reorder.
                size_t reorder_index = mkldnn_emitter->reserve_primitive_space(3);
                auto& deps = mkldnn_emitter->get_primitive_deps(reorder_index);
                auto functor =
                    [&, input_desc, result_desc, reorder_index, arg_buffer_index, out_buffer_index](
                        CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_reorder(ctx->mkldnn_memories,
                                                          ctx->mkldnn_primitives,
                                                          ctx->mkldnn_scratchpad_mds,
                                                          input_desc,
                                                          result_desc,
                                                          deps,
                                                          reorder_index);
                        }
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[out_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx, reorder_index, deps, cpu::mkldnn_utils::OpType::CONVERTLAYOUT);
                    };
                functors.emplace_back(functor);
            }

            void register_builders_convert_layout_cpp() { REGISTER_CPU_OP_BUILDER(ConvertLayout); }
        }
    }
}
