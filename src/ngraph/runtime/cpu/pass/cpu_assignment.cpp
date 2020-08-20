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

#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"
#include <algorithm>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include <dnnl.hpp>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/conv_fused.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/gelu.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/deconv.hpp"
#include "ngraph/runtime/cpu/op/gelu_backprop.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/quantized_matmul.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/update_slice.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v1::Add)
                {
                    (void)external_function;
                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto arg0_rank = arg0_shape.size();
                    auto arg1_rank = arg1_shape.size();

                    auto src_size = shape_size(arg0_shape);

                    // insert Add as DNNL op, only if the src_size is big. this is to avoid DNNL
                    // overhead
                    // for smaller tensor sizes
                    if (node->get_input_element_type(0) == element::f32 &&
                        node->get_input_element_type(1) == element::f32 && arg0_rank == 4 &&
                        arg1_rank == 4 && src_size > 64000)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Concat)
                {
                    (void)external_function;
                    if ((node->get_input_element_type(0) == element::f32 ||
                         node->get_input_element_type(0) == element::i8 ||
                         node->get_input_element_type(0) == element::u8) &&
                        ((node->get_input_shape(0)).size() == 4 ||
                         (node->get_input_shape(0)).size() == 2))
                    {
                        // DNNL seems to throw an exception when given tensors with 0-length
                        // dimensions, so don't assign it in such cases.
                        bool any_zero = false;

                        for (size_t i = 0; i < node->get_input_size(); i++)
                        {
                            if (shape_size(node->get_input_shape(i)) == 0)
                            {
                                any_zero = true;
                                break;
                            }
                        }

                        if (!any_zero)
                        {
                            runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                        }
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Convolution)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::Convolution>(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::GroupConvolution)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::GroupConvolution>(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::GroupConvolutionBias)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_conv<ngraph::op::GroupConvolutionBias>(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionRelu)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_conv<ngraph::op::ConvolutionRelu>(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ConvolutionBiasAdd)
                {
                    (void)external_function;
                    auto convolution = static_cast<ngraph::op::v0::ConvolutionBiasAdd*>(node);

                    if (dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::ConvolutionBiasAdd>(node))
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        const int ADD_INPUT = 3;
                        // Accumulates conv into the second input of the unfused add
                        op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionAdd)
                {
                    (void)external_function;
                    auto convolution = static_cast<ngraph::op::ConvolutionAdd*>(node);

                    if (dnnl_utils::can_use_dnnl_conv<ngraph::op::ConvolutionAdd>(node))
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        const int ADD_INPUT = 2;
                        // Accumulates conv into the second input of the unfused add
                        op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormInferenceRelu)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_batchnorm_fprop(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormTrainingRelu)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_batchnorm_fprop(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::DeconvolutionBias)
                {
                    (void)external_function;
                    auto convolution = static_cast<ngraph::op::DeconvolutionBias*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto arg2_shape = node->get_input_shape(2);
                    auto result_shape = node->get_output_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto arg1_rank = arg1_shape.size();
                    auto arg2_rank = arg2_shape.size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides_forward())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated &&
                        ((arg0_rank == 4 && arg1_rank == 4) ||
                         (arg0_rank == 5 && arg1_rank == 5)) &&
                        (arg2_rank == 1) && node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        convolution->set_op_annotations(op_annotations);
                    }
                    else
                    {
                        NGRAPH_DEBUG << "DeconvolutionBias : data_dilated = " << data_dilated;
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ConvolutionBackpropData)
                {
                    (void)external_function;
                    auto convolution = static_cast<ngraph::op::v0::ConvolutionBackpropData*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto result_shape = node->get_output_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto arg1_rank = arg1_shape.size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides_forward())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated &&
                        ((arg0_rank == 4 && arg1_rank == 4) ||
                         (arg0_rank == 5 && arg1_rank == 5)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ConvolutionBackpropFilters)
                {
                    (void)external_function;
                    auto convolution =
                        static_cast<ngraph::op::v0::ConvolutionBackpropFilters*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto result_shape = node->get_output_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto arg1_rank = arg1_shape.size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides_forward())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated &&
                        ((arg0_rank == 4 && arg1_rank == 4) ||
                         (arg0_rank == 5 && arg1_rank == 5)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ConvolutionBias)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_conv<ngraph::op::v0::ConvolutionBias>(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ConvolutionBiasBackpropFiltersBias)
                {
                    (void)external_function;
                    auto convolution =
                        static_cast<ngraph::op::v0::ConvolutionBiasBackpropFiltersBias*>(node);

                    auto data_shape = node->get_input_shape(0);
                    auto delta_shape = node->get_input_shape(1);
                    auto data_rank = data_shape.size();
                    auto delta_rank = delta_shape.size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides_forward())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated && data_rank == delta_rank &&
                        (data_rank == 4 || data_rank == 5) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::AvgPool)
                {
                    (void)external_function;
                    auto avg_pool = static_cast<ngraph::op::v0::AvgPool*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg0_rank == 4 && avg_pool->get_window_shape().size() == 2) ||
                         (arg0_rank == 5 && avg_pool->get_window_shape().size() == 3)) &&
                        (node->get_input_element_type(0) == element::f32 ||
                         node->get_input_element_type(0) == element::u8 ||
                         node->get_input_element_type(0) == element::i8))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::AvgPoolBackprop)
                {
                    (void)external_function;
                    auto avg_pool = static_cast<ngraph::op::v0::AvgPoolBackprop*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg0_rank == 4 && avg_pool->get_window_shape().size() == 2) ||
                         (arg0_rank == 5 && avg_pool->get_window_shape().size() == 3)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::MaxPool)
                {
                    (void)external_function;
                    auto max_pool = static_cast<ngraph::op::v0::MaxPool*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg0_rank == 4 && max_pool->get_window_shape().size() == 2) ||
                         (arg0_rank == 5 && max_pool->get_window_shape().size() == 3)) &&
                        (node->get_input_element_type(0) == element::f32 ||
                         node->get_input_element_type(0) == element::u8 ||
                         node->get_input_element_type(0) == element::i8 ||
                         (node->get_input_element_type(0) == element::bf16 &&
                          runtime::cpu::dnnl_utils::is_bf16_supported())))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPoolWithIndices)
                {
                    (void)external_function;
                    auto max_pool = static_cast<ngraph::op::MaxPoolWithIndices*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg0_rank == 4 && max_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::MaxPoolBackprop)
                {
                    (void)external_function;
                    auto max_pool = static_cast<ngraph::op::v0::MaxPoolBackprop*>(node);

                    auto arg1_shape = node->get_input_shape(1);
                    auto arg1_rank = arg1_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg1_rank == 4 && max_pool->get_window_shape().size() == 2) ||
                         (arg1_rank == 5 && max_pool->get_window_shape().size() == 3)) &&
                        node->get_input_element_type(1) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPoolWithIndicesBackprop)
                {
                    (void)external_function;
                    auto max_pool = static_cast<ngraph::op::MaxPoolWithIndicesBackprop*>(node);

                    auto arg1_shape = node->get_input_shape(1);
                    auto arg1_rank = arg1_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg1_rank == 4 && max_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(1) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Relu)
                {
                    (void)external_function;
                    auto relu = static_cast<ngraph::op::v0::Relu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 3 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        if (get_user_count(node->input_value(0)) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        relu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ReplaceSlice)
                {
                    (void)external_function;
                    auto replace_slice = static_cast<ngraph::op::v0::ReplaceSlice*>(node);

                    // ReplaceSlice is independent of data type. Hence not checking type
                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    if (get_user_count(node->input_value(0)) == 1)
                    {
                        // Safe to overwrite input
                        op_annotations->add_in_place_oi_pair({0, 0, true});
                    }
                    replace_slice->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::UpdateSlice)
                {
                    (void)external_function;
                    auto update_slice = static_cast<ngraph::op::UpdateSlice*>(node);

                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    if (get_user_count(node->input_value(0)) == 1)
                    {
                        // Safe to overwrite input
                        op_annotations->add_in_place_oi_pair({0, 0, true});
                    }
                    update_slice->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ScatterAdd)
                {
                    (void)external_function;
                    auto scatter_add = static_cast<ngraph::op::v0::ScatterAdd*>(node);

                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    if (get_user_count(node->input_value(0)) == 1)
                    {
                        // Safe to overwrite input
                        op_annotations->add_in_place_oi_pair({0, 0, true});
                    }
                    scatter_add->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::LRN)
                {
                    (void)external_function;
                    auto lrn = static_cast<ngraph::op::v0::LRN*>(node);
                    AxisSet axes = lrn->get_reduction_axes();
                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4) && node->get_input_element_type(0) == element::f32 &&
                        axes == AxisSet{1})
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Sigmoid)
                {
                    (void)external_function;
                    if (node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::SigmoidBackprop)
                {
                    (void)external_function;
                    if (node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::ReluBackprop)
                {
                    (void)external_function;
                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::BatchNormTraining)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_batchnorm_fprop(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::BatchNormInference)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_batchnorm_fprop(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::BatchNormTrainingBackprop)
                {
                    (void)external_function;
                    if (dnnl_utils::can_use_dnnl_batchnorm_bprop(node))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Lstm)
                {
                    (void)external_function;
                    auto src_layer_rank = node->get_input_shape(0).size();
                    auto src_iter_rank = node->get_input_shape(1).size();
                    auto src_iter_c_rank = node->get_input_shape(2).size();
                    auto weights_layer_rank = node->get_input_shape(3).size();
                    auto weights_iter_rank = node->get_input_shape(4).size();
                    auto bias_rank = node->get_input_shape(5).size();
                    if ((src_layer_rank == 2 && src_iter_rank == 2 && src_iter_c_rank == 2 &&
                         weights_layer_rank == 2 && weights_iter_rank == 2 && bias_rank == 1 &&
                         node->get_input_element_type(0) == element::f32 &&
                         node->get_input_element_type(1) == element::f32))
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Rnn)
                {
                    (void)external_function;
                    auto rnn_op = static_cast<ngraph::op::Rnn*>(node);
                    auto src_layer_rank = node->get_input_shape(0).size();
                    auto src_iter_rank = node->get_input_shape(1).size();

                    if (rnn_op->is_type(ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm))
                    {
                        auto src_iter_c_rank = node->get_input_shape(2).size();
                        auto weights_layer_rank = node->get_input_shape(3).size();
                        auto weights_iter_rank = node->get_input_shape(4).size();
                        auto bias_rank = node->get_input_shape(5).size();
                        if ((src_layer_rank == 2 && src_iter_rank == 2 && src_iter_c_rank == 2 &&
                             weights_layer_rank == 2 && weights_iter_rank == 2 && bias_rank == 1 &&
                             node->get_input_element_type(0) == element::f32 &&
                             node->get_input_element_type(1) == element::f32))
                        {
                            runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                        }
                    }
                    else if (rnn_op->is_type(ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_rnn))
                    {
                        auto weights_layer_rank = node->get_input_shape(2).size();
                        auto weights_iter_rank = node->get_input_shape(3).size();
                        auto bias_rank = node->get_input_shape(4).size();
                        if ((src_layer_rank == 2 && src_iter_rank == 2 && weights_layer_rank == 2 &&
                             weights_iter_rank == 2 && bias_rank == 1 &&
                             node->get_input_element_type(0) == element::f32 &&
                             node->get_input_element_type(1) == element::f32))
                        {
                            runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                        }
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Softmax)
                {
                    (void)external_function;
                    auto softmax = static_cast<ngraph::op::v0::Softmax*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32 &&
                        softmax->get_axes().size() == 1)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Slice)
                {
                    (void)external_function;
                    auto slice = static_cast<ngraph::op::v0::Slice*>(node);
                    auto strides = slice->get_strides();
                    if (!is_strided(strides) && node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BoundedRelu)
                {
                    (void)external_function;
                    auto bounded_relu = static_cast<ngraph::op::BoundedRelu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        if (get_user_count(node->input_value(0)) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        bounded_relu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Gelu)
                {
                    (void)external_function;
                    auto gelu = static_cast<ngraph::op::v0::Gelu*>(node);

                    if (node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                        if (get_user_count(node->input_value(0)) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        gelu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::GeluBackprop)
                {
                    (void)external_function;
                    auto gelu = static_cast<ngraph::op::GeluBackprop*>(node);

                    if (node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                        if (get_user_count(node->input_value(0)) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        gelu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::CPULeakyRelu)
                {
                    (void)external_function;
                    auto leaky_relu = static_cast<ngraph::op::CPULeakyRelu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_dnnl_op(true);
                        if (get_user_count(node->input_value(0)) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        leaky_relu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::QuantizedConvolution)
                {
                    (void)external_function;
                    auto qconv = static_cast<ngraph::op::v0::QuantizedConvolution*>(node);
                    auto input_zero_point =
                        as_type_ptr<ngraph::op::v0::Constant>(qconv->get_argument(3));
                    auto filter_zero_point =
                        as_type_ptr<ngraph::op::v0::Constant>(qconv->get_argument(5));
                    auto output_zero_point =
                        as_type_ptr<ngraph::op::v0::Constant>(qconv->get_argument(7));
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8)
                    {
                        // Mkldnn assumes zero point to be zero
                        if (input_zero_point == nullptr || filter_zero_point == nullptr ||
                            output_zero_point == nullptr || !(ngraph::is_zero(input_zero_point)) ||
                            !(ngraph::is_zero(filter_zero_point)) ||
                            !(ngraph::is_zero(output_zero_point)))
                        {
                            return;
                        }

                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::QuantizedConvolutionRelu)
                {
                    (void)external_function;
                    runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::QuantizedConvolutionBias)
                {
                    (void)external_function;
                    runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::QuantizedConvolutionBiasAdd)
                {
                    (void)external_function;
                    auto quantized_conv_bias =
                        static_cast<ngraph::op::v0::QuantizedConvolutionBiasAdd*>(node);
                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    op_annotations->set_dnnl_op(true);
                    const int ADD_INPUT = 3;
                    // Accumulates conv into the second input of the unfused add
                    op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                    quantized_conv_bias->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::QuantizedConvolutionBiasSignedAdd)
                {
                    (void)external_function;
                    auto quantized_conv_bias =
                        static_cast<ngraph::op::v0::QuantizedConvolutionBiasSignedAdd*>(node);
                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    op_annotations->set_dnnl_op(true);
                    const int ADD_INPUT = 3;
                    // Accumulates conv into the second input of the unfused add
                    op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                    quantized_conv_bias->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::QuantizedDotBias)
                {
                    (void)external_function;
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedMatmul)
                {
                    (void)external_function;
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8)
                    {
                        runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Dequantize)
                {
                    (void)external_function;
                    auto dequantize = static_cast<ngraph::op::v0::Dequantize*>(node);
                    // TODO(nbpatel): Support dynamic offset via dnnl
                    // Go through reference if the offset is not a constant
                    if (!dequantize->get_argument(2)->is_constant())
                    {
                        return;
                    }
                    auto offset_const_op = std::static_pointer_cast<ngraph::op::v0::Constant>(
                        dequantize->get_argument(2));
                    // TODO: DNNL only handles float / not double
                    if (node->get_output_element_type(0) != element::f32)
                    {
                        return;
                    }
                    if (node->get_input_element_type(0) == element::u8)
                    {
                        auto offset = offset_const_op->get_vector<uint8_t>();
                        if (offset[0] != 0)
                            return;
                    }
                    if (node->get_input_element_type(0) == element::i8)
                    {
                        auto offset = offset_const_op->get_vector<int8_t>();
                        if (offset[0] != 0)
                            return;
                    }
                    if (node->get_input_element_type(0) == element::i32)
                    {
                        auto offset = offset_const_op->get_vector<int32_t>();
                        if (offset[0] != 0)
                            return;
                    }
                    runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Quantize)
                {
                    (void)external_function;
                    auto quantize = static_cast<ngraph::op::v0::Quantize*>(node);
                    // TODO(nbpatel): Support dynamic offset via dnnl
                    // Go through reference if the offset is not a constant
                    if (!quantize->get_argument(2)->is_constant())
                    {
                        return;
                    }
                    auto offset_const_op = std::static_pointer_cast<ngraph::op::v0::Constant>(
                        quantize->get_argument(2));
                    ngraph::op::v0::Quantize::RoundMode round_mode = quantize->get_round_mode();
                    if (round_mode !=
                        ngraph::op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN)
                    {
                        return;
                    }
                    // TODO: DNNL only handles float / not double
                    if (node->get_input_element_type(0) != element::f32)
                    {
                        return;
                    }
                    if (node->get_output_element_type(0) == element::u8)
                    {
                        auto offset = offset_const_op->get_vector<uint8_t>();
                        if (offset[0] != 0)
                        {
                            return;
                        }
                    }
                    if (node->get_output_element_type(0) == element::i8)
                    {
                        auto offset = offset_const_op->get_vector<int8_t>();
                        if (offset[0] != 0)
                        {
                            return;
                        }
                    }
                    if (node->get_output_element_type(0) == element::i32)
                    {
                        auto offset = offset_const_op->get_vector<int32_t>();
                        if (offset[0] != 0)
                        {
                            return;
                        }
                    }
                    runtime::cpu::dnnl_utils::assign_dnnl_kernel(node);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::v0::Convert)
                {
                    (void)external_function;
                    auto convert = static_cast<ngraph::op::v0::Convert*>(node);
                    if ((node->get_input_element_type(0) == element::i8 &&
                         node->get_output_element_type(0) == element::u8) ||
                        (node->get_input_element_type(0) == element::u8 &&
                         node->get_output_element_type(0) == element::i8))
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->add_in_place_oi_pair({0, 0, false});
                        convert->set_op_annotations(op_annotations);
                    }
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::cpu::pass::AssignOpMap s_dispatcher{
    {TI(ngraph::op::v1::Add), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v1::Add>},
    {TI(ngraph::op::v0::Concat),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Concat>},
    {TI(ngraph::op::v0::Convert),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Convert>},
    {TI(ngraph::op::v0::AvgPool),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::AvgPool>},
    {TI(ngraph::op::v0::AvgPoolBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::AvgPoolBackprop>},
    {TI(ngraph::op::v0::BatchNormTraining),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::BatchNormTraining>},
    {TI(ngraph::op::v0::BatchNormInference),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::BatchNormInference>},
    {TI(ngraph::op::BoundedRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BoundedRelu>},
    {TI(ngraph::op::v0::BatchNormTrainingBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::BatchNormTrainingBackprop>},
    {TI(ngraph::op::v0::Convolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Convolution>},
    {TI(ngraph::op::v0::GroupConvolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::GroupConvolution>},
    {TI(ngraph::op::ConvolutionRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionRelu>},
    {TI(ngraph::op::v0::ConvolutionBiasAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::ConvolutionBiasAdd>},
    {TI(ngraph::op::BatchNormTrainingRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormTrainingRelu>},
    {TI(ngraph::op::BatchNormInferenceRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormInferenceRelu>},
    {TI(ngraph::op::v0::ConvolutionBackpropData),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::ConvolutionBackpropData>},
    {TI(ngraph::op::v0::ConvolutionBackpropFilters),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::ConvolutionBackpropFilters>},
    {TI(ngraph::op::v0::MaxPool),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::MaxPool>},
    {TI(ngraph::op::MaxPoolWithIndices),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::MaxPoolWithIndices>},
    {TI(ngraph::op::v0::MaxPoolBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::MaxPoolBackprop>},
    {TI(ngraph::op::MaxPoolWithIndicesBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::MaxPoolWithIndicesBackprop>},
    {TI(ngraph::op::v0::ConvolutionBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::ConvolutionBias>},
    {TI(ngraph::op::v0::QuantizedConvolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::QuantizedConvolution>},
    {TI(ngraph::op::v0::ConvolutionBiasBackpropFiltersBias),
     &runtime::cpu::pass::CPUAssignment::assign<
         ngraph::op::v0::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::op::v0::LRN), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::LRN>},
    {TI(ngraph::op::v0::Relu), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Relu>},
    {TI(ngraph::op::v0::ReluBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::ReluBackprop>},
    {TI(ngraph::op::CPULeakyRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::CPULeakyRelu>},
    {TI(ngraph::op::v0::Sigmoid),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Sigmoid>},
    {TI(ngraph::op::v0::SigmoidBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::SigmoidBackprop>},
    {TI(ngraph::op::Lstm), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Lstm>},
    {TI(ngraph::op::Rnn), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Rnn>},
    {TI(ngraph::op::v0::Softmax),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Softmax>},
    {TI(ngraph::op::v0::Slice), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Slice>},
    {TI(ngraph::op::v0::ReplaceSlice),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::ReplaceSlice>},
    {TI(ngraph::op::UpdateSlice),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::UpdateSlice>},
    {TI(ngraph::op::ConvolutionAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionAdd>},
    {TI(ngraph::op::v0::QuantizedConvolutionRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::QuantizedConvolutionRelu>},
    {TI(ngraph::op::v0::QuantizedConvolutionBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::QuantizedConvolutionBias>},
    {TI(ngraph::op::v0::QuantizedConvolutionBiasAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::QuantizedConvolutionBiasAdd>},
    {TI(ngraph::op::v0::QuantizedConvolutionBiasSignedAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::QuantizedConvolutionBiasSignedAdd>},
    {TI(ngraph::op::GroupConvolutionBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::GroupConvolutionBias>},
    {TI(ngraph::op::v0::Quantize),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Quantize>},
    {TI(ngraph::op::v0::Dequantize),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Dequantize>},
    {TI(ngraph::op::QuantizedMatmul),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedMatmul>},
    {TI(ngraph::op::v0::QuantizedDotBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::QuantizedDotBias>},
    {TI(ngraph::op::DeconvolutionBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::DeconvolutionBias>},
    {TI(ngraph::op::v0::ScatterAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::ScatterAdd>},
    {TI(ngraph::op::v0::Gelu), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::v0::Gelu>},
    {TI(ngraph::op::GeluBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::GeluBackprop>},
};

bool runtime::cpu::pass::CPUAssignment::run_on_call_graph(
    const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& node : nodes)
    {
        auto& n = *node;
        auto handler = s_dispatcher.find(TI(n));
        if (handler != s_dispatcher.end())
        {
            handler->second(m_external_function, node.get());
        }
    }

    return false;
}
