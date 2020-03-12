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
#include <algorithm>
#include <iterator>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/fused/normalize_l2.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::NormalizeL2::type_info;

op::NormalizeL2::NormalizeL2(const Output<Node>& data,
                             const Output<Node>& axes,
                             float eps,
                             EpsMode eps_mode)
    : FusedOp({data, axes})
    , m_eps{eps}
    , m_eps_mode{eps_mode}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::NormalizeL2::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("eps_mode", m_eps_mode);
    return true;
}

void op::NormalizeL2::pre_validate_and_infer_types()
{
    auto axes_node = input_value(1).get_node_shared_ptr();
    const auto& input_pshape = get_input_partial_shape(0);
    const auto& axes_pshape = get_input_partial_shape(1);
    const auto& input_rank = input_pshape.rank();
    const auto& axes_rank = axes_pshape.rank();

    NODE_VALIDATION_CHECK(this, axes_node->is_constant(), "Input axes must be Constant type");

    if (axes_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              axes_rank.get_length() <= 1,
                              "Input axes must be scalar or have rank equal to 1 (axes rank: ",
                              axes_rank,
                              ").");

        if (input_rank.is_static())
        {
            const auto reduction_axes = get_reduction_axes();
            for (auto axis : reduction_axes)
            {
                NODE_VALIDATION_CHECK(this,
                                      axis < input_rank.get_length(),
                                      "Reduction axis (",
                                      axis,
                                      ") is out of bounds ",
                                      "(argument shape: ",
                                      input_pshape,
                                      ")");
            }
        }
    }
}

AxisSet op::NormalizeL2::get_reduction_axes() const
{
    AxisSet axes;
    auto axes_input_node = input_value(1).get_node_shared_ptr();
    if (auto const_op = as_type_ptr<op::Constant>(axes_input_node))
    {
        axes = const_op->get_axis_set_val();
    }
    return axes;
}

NodeVector op::NormalizeL2::decompose_op() const
{
    Output<Node> data{input_value(0)};
    const Shape input_shape{data.get_shape()};

    AxisSet reduction_axes = get_reduction_axes();

    // Calculate l2 norm across axes determined by axes input
    auto builder_bias_mode =
        (m_eps_mode == EpsMode::MAX) ? builder::BiasMode::MAX : builder::BiasMode::ADD;
    Output<Node> norm =
        builder::opset1::l2_norm(data, reduction_axes, m_eps, builder_bias_mode, true);

    data = make_shared<op::Divide>(data, norm, AutoBroadcastSpec(AutoBroadcastType::NUMPY));

    return as_node_vector({data});
}

shared_ptr<Node> op::NormalizeL2::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<NormalizeL2>(new_args.at(0), new_args.at(1), m_eps, m_eps_mode);
}
