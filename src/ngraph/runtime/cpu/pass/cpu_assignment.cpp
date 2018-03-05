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
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

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

                    auto src_size = 1;
                    for (size_t i = 0; i < node->get_input_shape(0).size(); i++)
                    {
                        src_size *= arg0_shape[0];
                    }
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
                void CPUAssignment::ASSIGN_DECL(ngraph::op::Relu)
                {
                    auto avg_pool = static_cast<op::Relu*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg0_rank == 4 && node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        avg_pool->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void CPUAssignment::ASSIGN_DECL(ngraph::op::ReluBackprop)
                {
                    auto avg_pool = static_cast<op::ReluBackprop*>(node);

                    auto arg0_shape = node->get_input_shape(0);
                    auto arg0_rank = arg0_shape.size();
                    auto result_shape = node->get_output_shape(0);

                    if (arg0_rank == 4 && node->get_input_element_type(0) == element::f32)
                    {
                        auto op_annotations =
                            std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                        op_annotations->set_mkldnn_op(true);
                        avg_pool->set_op_annotations(op_annotations);
                    }
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::cpu::pass::AssignOpMap s_dispatcher{
    {TI(ngraph::op::Add), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Add>},
    {TI(ngraph::op::Convolution),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Convolution>},
    {TI(ngraph::op::ConvolutionBackpropData),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBackpropData>},
    {TI(ngraph::op::ConvolutionBackpropFilters),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ConvolutionBackpropFilters>},
    {TI(ngraph::op::AvgPool), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::AvgPool>},
    {TI(ngraph::op::AvgPoolBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::AvgPoolBackprop>},
    {TI(ngraph::op::Relu), &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::Relu>},
    {TI(ngraph::op::ReluBackprop),
     &runtime::cpu::pass::CPUAssignment::assign<ngraph::op::ReluBackprop>},
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
