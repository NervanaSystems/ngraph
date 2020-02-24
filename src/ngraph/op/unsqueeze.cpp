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
#include <cstddef>
#include <functional>
#include <set>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Unsqueeze::type_info;

op::Unsqueeze::Unsqueeze(const Output<Node>& data, const Output<Node>& axes)
    : FusedOp({data, axes})
{
    constructor_validate_and_infer_types();
}

void op::Unsqueeze::pre_validate_and_infer_types()
{
    auto data = input_value(0);
    auto axes_node = input_value(1).get_node_shared_ptr();

    if (data.get_partial_shape().rank().is_dynamic() || !axes_node->is_constant())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        return;
    }

    // Get value of axes from Constant
    auto axes_constant = as_type_ptr<op::Constant>(axes_node);
    auto axes = axes_constant->cast_vector<size_t>();

    NODE_VALIDATION_CHECK(this, !axes.empty(), "'axes' input is mandatory.");
    NODE_VALIDATION_CHECK(this,
                          axes.size() == set<int64_t>(begin(axes), end(axes)).size(),
                          "'axes' input has a duplicate axis.");

    if (data.get_partial_shape().is_dynamic())
    {
        set_output_type(0,
                        get_input_element_type(0),
                        PartialShape::dynamic(data.get_partial_shape().rank() + axes.size()));
        return;
    }

    auto data_shape = data.get_shape();

    sort(begin(axes), end(axes), less<int64_t>());

    AxisVector input_order{ngraph::get_default_order(data_shape.size())};

    for (auto axis : axes)
    {
        NODE_VALIDATION_CHECK(
            this, axis <= data_shape.size(), "provided 'axes' value ", axis, " is not valid.");

        data_shape.insert(next(begin(data_shape), axis), 1);
    }
    set_output_type(0, get_input_element_type(0), data_shape);
}

NodeVector op::Unsqueeze::decompose_op() const
{
    NODE_VALIDATION_CHECK(
        this,
        (get_output_partial_shape(0).is_static()),
        "output shape was not calculated during pre_validate_and_infer_types. Can not decompose.");
    auto data = input_value(0);
    auto data_shape = data.get_shape();
    auto output_shape = get_output_shape(0);
    AxisVector input_order{ngraph::get_default_order(data_shape.size())};
    return {make_shared<ngraph::op::Reshape>(data, input_order, output_shape)};
}

shared_ptr<Node> op::Unsqueeze::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Unsqueeze>(new_args.at(0), new_args.at(1));
}
