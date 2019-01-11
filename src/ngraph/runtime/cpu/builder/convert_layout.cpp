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

#include "ngraph/runtime/cpu/op/convert_layout.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"

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

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();

                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 0);
                auto result_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                if (input_desc.data.format == mkldnn_nchw &&
                    result_desc.data.format == mkldnn_goihw)
                {
                    //becomes a copy
                    input_desc = result_desc;
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
                // handle special case for rnn formats
                if (auto rnn_back_prop =
                        std::dynamic_pointer_cast<ngraph::op::RnnBackprop>(node->get_users()[0]))
                {
                    auto rnn_attributes = rnn_back_prop->get_rnn_attributes();
                    auto create_mkldnn_data_desc = [&](Shape& input_shape,
                                                       mkldnn::memory::format layout) {
                        input_desc = mkldnn::memory::desc(
                            mkldnn::memory::dims(input_shape.begin(), input_shape.end()),
                            mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                            layout);
                    };

                    if (result_desc.data.format == mkldnn_tnc && input_desc.data.ndims == 2)
                    {
                        Shape input_shape{rnn_attributes.timestep,
                                          rnn_attributes.batch,
                                          rnn_attributes.feature_size};
                        create_mkldnn_data_desc(input_shape, mkldnn::memory::format::tnc);
                    }

                    if (result_desc.data.format == mkldnn_ldsnc && input_desc.data.ndims == 2)
                    {
                        Shape input_shape{rnn_attributes.layer,
                                          rnn_attributes.direction,
                                          rnn_attributes.states,
                                          rnn_attributes.batch,
                                          rnn_attributes.feature_size};
                        create_mkldnn_data_desc(input_shape, mkldnn::memory::format::ldsnc);
                    }

                    if (result_desc.data.format == mkldnn_ldigo && input_desc.data.ndims == 2)
                    {
                        Shape input_shape{rnn_attributes.layer,
                                          rnn_attributes.direction,
                                          rnn_attributes.gates,
                                          rnn_attributes.dlc,
                                          rnn_attributes.slc};
                        create_mkldnn_data_desc(input_shape, mkldnn::memory::format::ldgoi);
                    }

                    if (result_desc.data.format == mkldnn_ldgoi && input_desc.data.ndims == 2)
                    {
                        Shape input_shape{rnn_attributes.layer,
                                          rnn_attributes.direction,
                                          rnn_attributes.gates,
                                          rnn_attributes.dlc,
                                          rnn_attributes.slc};
                        create_mkldnn_data_desc(input_shape, mkldnn::memory::format::ldigo);
                    }

                    if (result_desc.data.format == mkldnn_ldgo && input_desc.data.ndims == 1)
                    {
                        Shape input_shape{rnn_attributes.layer,
                                          rnn_attributes.direction,
                                          rnn_attributes.gates,
                                          rnn_attributes.sic};
                        create_mkldnn_data_desc(input_shape, mkldnn::memory::format::ldgo);
                    }
                }
                std::cout << "input_desc: " << input_desc.data.format
                          << " ndims: " << input_desc.data.ndims
                          << "   result_desc: " << result_desc.data.format
                          << " ndims: " << result_desc.data.ndims << std::endl;

                size_t reorder_index = mkldnn_emitter->build_reorder(input_desc, result_desc);

                auto& deps = mkldnn_emitter->get_primitive_deps(reorder_index);
                auto functor = [&, reorder_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, reorder_index);
                };
                functors.emplace_back(functor);
            }
        }
    }
}
