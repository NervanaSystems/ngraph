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
#include "ngraph/ops/convolution.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

static const runtime::cpu::pass::AssignOpMap dispatcher{
    {TI(ngraph::op::Convolution), &runtime::cpu::pass::CPUAssignment::AssignConvolution},
};

bool runtime::cpu::pass::CPUAssignment::run_on_call_graph(
    const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& node : nodes)
    {
        auto& n = *node;
        auto handler = dispatcher.find(TI(n));
        if (handler != dispatcher.end())
        {
            handler->second(m_external_function.get(), node.get());
        }
    }

    return false;
}

void runtime::cpu::pass::CPUAssignment::ASSIGN_DECL(AssignConvolution)
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
        auto op_annotations = std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
        op_annotations->set_mkldnn_op(true);
        convolution->set_op_annotations(op_annotations);
    }
}
