/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/max_pool_with_indices.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"

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
                    auto add = static_cast<op::Add*>(node);
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
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        add->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Concat)
                {
                    auto concat = static_cast<op::Concat*>(node);

                    if (node->get_input_element_type(0) == element::f32 &&
                        ((node->get_input_shape(0)).size() == 4 ||
                         (node->get_input_shape(0)).size() == 2))
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        concat->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Convolution)
                {
                    auto convolution = static_cast<op::Convolution*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg1_shape = node->get_input_shape(1);
                    auto result_shape = node->get_output_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto arg1_rank = arg1_shape.size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated && arg0_rank == 4 && arg1_rank == 4 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionRelu)
                {
                    auto convolution = static_cast<op::ConvolutionRelu*>(node);

                    auto arg0_rank = node->get_input_shape(0).size();
                    auto arg1_rank = node->get_input_shape(1).size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated && arg0_rank == 4 && arg1_rank == 4 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionBiasRelu)
                {
                    auto convolution = static_cast<op::ConvolutionBiasRelu*>(node);

                    auto arg0_rank = node->get_input_shape(0).size();
                    auto arg1_rank = node->get_input_shape(1).size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated && arg0_rank == 4 && arg1_rank == 4 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormRelu)
                {
                    if (node->get_argument(2 /*input data*/)->get_shape().size() == 4)
                    {
                        auto bn_relu = static_cast<op::BatchNormRelu*>(node);
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        bn_relu->set_op_annotations(op_annotations);
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

                    if (!data_dilated && arg0_rank == 4 && arg1_rank == 4 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        convolution->set_op_annotations(op_annotations);
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

                    if (!data_dilated && arg0_rank == 4 && arg1_rank == 4 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ConvolutionBias)
                {
                    auto convolution = static_cast<op::ConvolutionBias*>(node);

                    auto data_shape = node->get_input_shape(0);
                    auto weights_shape = node->get_input_shape(1);
                    auto result_shape = node->get_output_shape(0);
                    auto data_rank = data_shape.size();
                    auto weights_rank = weights_shape.size();

                    bool data_dilated = false;
                    for (size_t s : convolution->get_data_dilation_strides())
                    {
                        data_dilated = data_dilated || (s != 1);
                    }

                    if (!data_dilated && data_rank == 4 && weights_rank == 4 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        convolution->set_op_annotations(op_annotations);
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

                    if (!data_dilated && data_rank == 4 && delta_rank == 4 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        convolution->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::AvgPool)
                {
                    auto avg_pool = static_cast<op::AvgPool*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg0_rank == 4 && avg_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        avg_pool->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::AvgPoolBackprop)
                {
                    auto avg_pool = static_cast<op::AvgPoolBackprop*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg0_rank == 4 && avg_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        avg_pool->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPool)
                {
                    auto max_pool = static_cast<op::MaxPool*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg0_rank == 4 && max_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        max_pool->set_op_annotations(op_annotations);
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
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        max_pool->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::MaxPoolBackprop)
                {
                    auto max_pool = static_cast<op::MaxPoolBackprop*>(node);

                    auto arg1_shape = node->get_input_shape(1);
                    auto arg1_rank = arg1_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg1_rank == 4 && max_pool->get_window_shape().size() == 2 &&
                        node->get_input_element_type(1) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        max_pool->set_op_annotations(op_annotations);
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
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        max_pool->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Relu)
                {
                    auto relu = static_cast<op::Relu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        relu->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Sigmoid)
                {
                    auto sigmoid = static_cast<op::Sigmoid*>(node);
                    if (node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        sigmoid->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::SigmoidBackprop)
                {
                    auto sigmoid = static_cast<op::SigmoidBackprop*>(node);
                    if (node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        sigmoid->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ReluBackprop)
                {
                    auto relu_bprop = static_cast<op::ReluBackprop*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if ((arg0_rank == 4 || arg0_rank == 2) &&
                        node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        relu_bprop->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNorm)
                {
                    auto input_shape = node->get_input_shape(2);
                    auto input_rank = input_shape.size();
                    if ((input_rank == 4 && node->get_input_element_type(2) == element::f32))
                    {
                        auto batchnorm = static_cast<op::BatchNorm*>(node);
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        batchnorm->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::BatchNormBackprop)
                {
                    auto input_shape = node->get_input_shape(2);
                    auto input_rank = input_shape.size();
                    auto delta_shape = node->get_input_shape(5);
                    auto delta_rank = delta_shape.size();
                    if ((input_rank == 4 && delta_rank == 4 &&
                         node->get_input_element_type(5) == element::f32 &&
                         node->get_input_element_type(2) == element::f32))
                    {
                        auto batchnorm = static_cast<op::BatchNormBackprop*>(node);
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        batchnorm->set_op_annotations(op_annotations);
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
                        auto rnn_node = static_cast<op::Rnn*>(node);
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        rnn_node->set_op_annotations(op_annotations);
                    }
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
    {TI(ngraph::op::BatchNorm), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNorm>},
    {TI(ngraph::op::BatchNormBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormBackprop>},
    {TI(ngraph::op::Convolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Convolution>},
    {TI(ngraph::op::ConvolutionRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionRelu>},
    {TI(ngraph::op::ConvolutionBiasRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBiasRelu>},
    {TI(ngraph::op::BatchNormRelu),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::BatchNormRelu>},
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
    {TI(ngraph::op::ConvolutionBiasBackpropFiltersBias),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBiasBackpropFiltersBias>},
    {TI(ngraph::op::Relu), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Relu>},
    {TI(ngraph::op::ReluBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ReluBackprop>},
    {TI(ngraph::op::Sigmoid), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Sigmoid>},
    {TI(ngraph::op::SigmoidBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::SigmoidBackprop>},
    {TI(ngraph::op::Rnn), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Rnn>},
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
