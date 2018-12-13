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

#include "ngraph/runtime/cpu/pass/cpu_assignment.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include <mkldnn.hpp>

#include "ngraph/descriptor/output.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_add.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/group_conv.hpp"
#include "ngraph/runtime/cpu/op/group_conv_bias.hpp"
#include "ngraph/runtime/cpu/op/leaky_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
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
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Add)
                {
                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto arg0_rank = arg0_shape.size();
                    auto arg1_rank = arg1_shape.size();

                    auto src_size = shape_size(arg0_shape);

                    // insert Add as MKLDNN op, only if the src_size is big. this is to avoid MKLDNN overhead
                    // for smaller tensor sizes
                    if (node->get_input_element_type(0) == element::f32 &&
                        node->get_input_element_type(1) == element::f32 && arg0_rank == 4 &&
                        arg1_rank == 4 && src_size > 64000)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Concat)
                {
                    if (node->get_input_element_type(0) == element::f32 &&
                        ((node->get_input_shape(0)).size() == 4 ||
                         (node->get_input_shape(0)).size() == 2))
                    {
                        // MKLDNN seems to throw an exception when given tensors with 0-length
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
                            runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                        }
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Convolution)
                {
                    if (mkldnn_utils::can_use_mkldnn_conv<ngraph::op::Convolution>(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::GroupConvolution)
                {
                    if (mkldnn_utils::can_use_mkldnn_conv<ngraph::op::GroupConvolution>(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::GroupConvolutionBias)
                {
                    if (mkldnn_utils::can_use_mkldnn_conv<ngraph::op::GroupConvolutionBias>(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionRelu)
                {
                    if (mkldnn_utils::can_use_mkldnn_conv<ngraph::op::ConvolutionRelu>(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionBiasAdd)
                {
                    auto convolution = static_cast<op::ConvolutionBiasAdd*>(node);

                    if (mkldnn_utils::can_use_mkldnn_conv<ngraph::op::ConvolutionBiasAdd>(node))
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        const int ADD_INPUT = 3;
                        // Accumulates conv into the second input of the unfused add
                        op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::GetOutputElement)
                {
                    auto goe = static_cast<op::GetOutputElement*>(node);
                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    op_annotations->add_in_place_oi_pair({0, goe->get_n(), false});
                    goe->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionAdd)
                {
                    auto convolution = static_cast<op::ConvolutionAdd*>(node);

                    if (mkldnn_utils::can_use_mkldnn_conv<ngraph::op::ConvolutionAdd>(node))
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        const int ADD_INPUT = 2;
                        // Accumulates conv into the second input of the unfused add
                        op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormInferenceRelu)
                {
                    if (mkldnn_utils::can_use_mkldnn_batchnorm_fprop(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormTrainingRelu)
                {
                    if (mkldnn_utils::can_use_mkldnn_batchnorm_fprop(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionBackpropData)
                {
                    auto convolution = static_cast<op::ConvolutionBackpropData*>(node);

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

                    if (!data_dilated && ((arg0_rank == 4 && arg1_rank == 4) ||
                                          (arg0_rank == 5 && arg1_rank == 5)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionBackpropFilters)
                {
                    auto convolution = static_cast<op::ConvolutionBackpropFilters*>(node);

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

                    if (!data_dilated && ((arg0_rank == 4 && arg1_rank == 4) ||
                                          (arg0_rank == 5 && arg1_rank == 5)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionBias)
                {
                    if (mkldnn_utils::can_use_mkldnn_conv<ngraph::op::ConvolutionBias>(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionBiasBackpropFiltersBias)
                {
                    auto convolution = static_cast<op::ConvolutionBiasBackpropFiltersBias*>(node);

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
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::AvgPool)
                {
                    auto avg_pool = static_cast<op::AvgPool*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg0_rank == 4 && avg_pool->get_window_shape().size() == 2) ||
                         (arg0_rank == 5 && avg_pool->get_window_shape().size() == 3)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::AvgPoolBackprop)
                {
                    auto avg_pool = static_cast<op::AvgPoolBackprop*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg0_rank == 4 && avg_pool->get_window_shape().size() == 2) ||
                         (arg0_rank == 5 && avg_pool->get_window_shape().size() == 3)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPool)
                {
                    auto max_pool = static_cast<op::MaxPool*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg0_rank == 4 && max_pool->get_window_shape().size() == 2) ||
                         (arg0_rank == 5 && max_pool->get_window_shape().size() == 3)) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPoolWithIndices)
                {
                    auto max_pool = static_cast<op::MaxPoolWithIndices*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg0_rank == 4 && max_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPoolBackprop)
                {
                    auto max_pool = static_cast<op::MaxPoolBackprop*>(node);

                    auto arg1_shape = node->get_input_shape(1);
                    auto arg1_rank = arg1_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (((arg1_rank == 4 && max_pool->get_window_shape().size() == 2) ||
                         (arg1_rank == 5 && max_pool->get_window_shape().size() == 3)) &&
                        node->get_input_element_type(1) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPoolWithIndicesBackprop)
                {
                    auto max_pool = static_cast<op::MaxPoolWithIndicesBackprop*>(node);

                    auto arg1_shape = node->get_input_shape(1);
                    auto arg1_rank = arg1_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg1_rank == 4 && max_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(1) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Relu)
                {
                    auto relu = static_cast<op::Relu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 3 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        if (get_user_count(node->get_argument(0).get()) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        relu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ReplaceSlice)
                {
                    auto replace_slice = static_cast<op::ReplaceSlice*>(node);

                    // ReplaceSlice is independent of data type. Hence not checking type
                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    if (get_user_count(node->get_argument(0).get()) == 1)
                    {
                        // Safe to overwrite input
                        op_annotations->add_in_place_oi_pair({0, 0, true});
                    }
                    replace_slice->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::UpdateSlice)
                {
                    auto update_slice = static_cast<op::UpdateSlice*>(node);

                    auto op_annotations =
                        std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    if (get_user_count(node->get_argument(0).get()) == 1)
                    {
                        // Safe to overwrite input
                        op_annotations->add_in_place_oi_pair({0, 0, true});
                    }
                    update_slice->set_op_annotations(op_annotations);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::LRN)
                {
                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4) && node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Sigmoid)
                {
                    if (node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::SigmoidBackprop)
                {
                    if (node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ReluBackprop)
                {
                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormTraining)
                {
                    if (mkldnn_utils::can_use_mkldnn_batchnorm_fprop(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormInference)
                {
                    if (mkldnn_utils::can_use_mkldnn_batchnorm_fprop(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormTrainingBackprop)
                {
                    if (mkldnn_utils::can_use_mkldnn_batchnorm_bprop(node))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Lstm)
                {
                    auto src_layer_rank = node->get_input_shape(0).size();
                    auto src_iter_rank = node->get_input_shape(1).size();
                    auto weights_layer_rank = node->get_input_shape(2).size();
                    auto weights_iter_rank = node->get_input_shape(3).size();
                    auto bias_rank = node->get_input_shape(4).size();
                    if ((src_layer_rank == 2 && src_iter_rank == 2 && weights_layer_rank == 2 &&
                         weights_iter_rank == 2 && bias_rank == 1 &&
                         node->get_input_element_type(0) == element::f32 &&
                         node->get_input_element_type(1) == element::f32))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Rnn)
                {
                    auto src_layer_rank = node->get_input_shape(0).size();
                    auto src_iter_rank = node->get_input_shape(1).size();
                    auto weights_layer_rank = node->get_input_shape(2).size();
                    auto weights_iter_rank = node->get_input_shape(3).size();
                    auto bias_rank = node->get_input_shape(4).size();
                    if ((src_layer_rank == 2 && src_iter_rank == 2 && weights_layer_rank == 2 &&
                         weights_iter_rank == 2 && bias_rank == 1 &&
                         node->get_input_element_type(0) == element::f32 &&
                         node->get_input_element_type(1) == element::f32))
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Softmax)
                {
                    auto softmax = static_cast<op::Softmax*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32 &&
                        softmax->get_axes().size() == 1)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Slice)
                {
                    auto slice = static_cast<op::Slice*>(node);
                    auto strides = slice->get_strides();
                    if (!is_strided(strides) && node->get_input_element_type(0) == element::f32)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedMaxPool)
                {
                    if (node->get_input_element_type(0) == element::u8 ||
                        node->get_input_element_type(0) == element::i8)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedAvgPool)
                {
                    if (node->get_input_element_type(0) == element::u8 ||
                        node->get_input_element_type(0) == element::i8)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BoundedRelu)
                {
                    auto bounded_relu = static_cast<op::BoundedRelu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        if (get_user_count(node->get_argument(0).get()) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        bounded_relu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::LeakyRelu)
                {
                    auto leaky_relu = static_cast<op::LeakyRelu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        if (get_user_count(node->get_argument(0).get()) == 1)
                        {
                            // Safe to overwrite input
                            op_annotations->add_in_place_oi_pair({0, 0, true});
                        }
                        leaky_relu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedConvolution)
                {
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedConvolutionRelu)
                {
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedConvolutionBias)
                {
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8)
                    {
                        runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedConvolutionBiasAdd)
                {
                    auto quantized_conv_bias = static_cast<op::QuantizedConvolutionBiasAdd*>(node);
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8 &&
                        node->get_input_element_type(3) == element::u8)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        const int ADD_INPUT = 3;
                        // Accumulates conv into the second input of the unfused add
                        op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                        quantized_conv_bias->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::QuantizedConvolutionBiasSignedAdd)
                {
                    auto quantized_conv_bias =
                        static_cast<op::QuantizedConvolutionBiasSignedAdd*>(node);
                    if (node->get_input_element_type(0) == element::u8 &&
                        node->get_input_element_type(1) == element::i8 &&
                        node->get_input_element_type(3) == element::i8)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        const int ADD_INPUT = 3;
                        // Accumulates conv into the second input of the unfused add
                        op_annotations->add_in_place_oi_pair({0, ADD_INPUT, true});
                        quantized_conv_bias->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Dequantize)
                {
                    auto dequantize = static_cast<op::Dequantize*>(node);
                    auto offset_const_op =
                        std::static_pointer_cast<ngraph::op::Constant>(dequantize->get_argument(2));
                    // TODO: MKLDNN only handles float / not double
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
                    runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Quantize)
                {
                    auto quantize = static_cast<op::Quantize*>(node);
                    auto offset_const_op =
                        std::static_pointer_cast<ngraph::op::Constant>(quantize->get_argument(2));
                    op::Quantize::RoundMode round_mode = quantize->get_round_mode();
                    if (round_mode != op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN)
                    {
                        return;
                    }
                    // TODO: MKLDNN only handles float / not double
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
                    runtime::cpu::mkldnn_utils::assign_mkldnn_kernel(node);
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::cpu::pass::AssignOpMap s_dispatcher{
    {TI(ngraph::op::Add), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Add>},
    {TI(ngraph::op::Concat), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Concat>},
    {TI(ngraph::op::AvgPool), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::AvgPoolBackprop>},
    {TI(ngraph::op::BatchNormTraining),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormTraining>},
    {TI(ngraph::op::BatchNormInference),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormInference>},
    {TI(ngraph::op::BoundedRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BoundedRelu>},
    {TI(ngraph::op::BatchNormTrainingBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormTrainingBackprop>},
    {TI(ngraph::op::Convolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Convolution>},
    {TI(ngraph::op::GroupConvolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::GroupConvolution>},
    {TI(ngraph::op::ConvolutionRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionRelu>},
    {TI(ngraph::op::ConvolutionBiasAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBiasAdd>},
    {TI(ngraph::op::BatchNormTrainingRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormTrainingRelu>},
    {TI(ngraph::op::BatchNormInferenceRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormInferenceRelu>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBackpropData>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::MaxPool), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::MaxPool>},
    {TI(ngraph::op::MaxPoolWithIndices),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::MaxPoolWithIndices>},
    {TI(ngraph::op::MaxPoolBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::MaxPoolBackprop>},
    {TI(ngraph::op::MaxPoolWithIndicesBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::MaxPoolWithIndicesBackprop>},
    {TI(ngraph::op::ConvolutionBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBias>},
    {TI(ngraph::op::QuantizedConvolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedConvolution>},
    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::op::LRN), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::LRN>},
    {TI(ngraph::op::Relu), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Relu>},
    {TI(ngraph::op::ReluBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ReluBackprop>},
    {TI(ngraph::op::LeakyRelu), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::LeakyRelu>},
    {TI(ngraph::op::Sigmoid), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Sigmoid>},
    {TI(ngraph::op::SigmoidBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::SigmoidBackprop>},
    {TI(ngraph::op::Lstm), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Lstm>},
    {TI(ngraph::op::Rnn), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Rnn>},
    {TI(ngraph::op::QuantizedMaxPool),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedMaxPool>},
    {TI(ngraph::op::QuantizedAvgPool),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedAvgPool>},
    {TI(ngraph::op::Softmax), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Softmax>},
    {TI(ngraph::op::Slice), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Slice>},
    {TI(ngraph::op::ReplaceSlice),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ReplaceSlice>},
    {TI(ngraph::op::UpdateSlice),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::UpdateSlice>},
    {TI(ngraph::op::ConvolutionAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionAdd>},
    {TI(ngraph::op::QuantizedConvolutionRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedConvolutionRelu>},
    {TI(ngraph::op::QuantizedConvolutionBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedConvolutionBias>},
    {TI(ngraph::op::QuantizedConvolutionBiasAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedConvolutionBiasAdd>},
    {TI(ngraph::op::QuantizedConvolutionBiasSignedAdd),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::QuantizedConvolutionBiasSignedAdd>},
    {TI(ngraph::op::GroupConvolutionBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::GroupConvolutionBias>},
    {TI(ngraph::op::Quantize), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Quantize>},
    {TI(ngraph::op::Dequantize),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Dequantize>},
    {TI(ngraph::op::GetOutputElement),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::GetOutputElement>},
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
