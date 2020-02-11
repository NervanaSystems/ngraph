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
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Squeeze::type_info;

op::Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes)
    : FusedOp({data, axes})
{
    constructor_validate_and_infer_types();
}

void op::Squeeze::pre_validate_and_infer_types()
{
    auto data = input_value(0);
    auto axes_node = input_value(1).get_node_shared_ptr();

    if (data.get_partial_shape().is_dynamic() || !axes_node->is_constant())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        return;
    }

    // Get value of axes from Constant
    auto axes_constant = as_type_ptr<op::Constant>(axes_node);
    auto axes = axes_constant->cast_vector<size_t>();
    auto data_shape = data.get_shape();
    std::vector<uint64_t> axes_to_squeeze(data_shape.size());

    // Prepare set of unique axes marked to be removed from input data.
    if (axes.empty())
    {
        // Default behaviour is to remove all single dimension axes.
        for (size_t idx = 0; idx < data_shape.size(); ++idx)
        {
            if (data_shape.at(idx) == 1)
            {
                axes_to_squeeze.at(idx) = 1;
            }
            else
            {
                axes_to_squeeze.at(idx) = 0;
            }
        }
    }
    else
    {
        set<size_t, greater<size_t>> unique_axes(begin(axes), end(axes));
        for (uint64_t axis : unique_axes)
        {
            NODE_VALIDATION_CHECK(
                this,
                (data_shape.at(axis) == 1),
                "provided axis value is invalid. Only axes of size 1 may be removed.");
            axes_to_squeeze.at(axis) = 1;
        }
    }

    Shape output_data_shape;
    for (size_t idx = 0; idx < data_shape.size(); ++idx)
    {
        if (axes_to_squeeze.at(idx) == 0)
        {
            output_data_shape.push_back(data_shape.at(idx));
        }
    }

    set_output_type(0, get_input_element_type(0), output_data_shape);
}

NodeVector op::Squeeze::decompose_op() const
{
    NODE_VALIDATION_CHECK(
        this,
        (get_output_partial_shape(0).is_static()),
        "output shape was not calculated during pre_validate_and_infer_types. Can not decompose.");
    auto data = input_value(0);
    auto data_shape = data.get_shape();
    auto output_data_shape = get_output_shape(0);
    AxisVector input_order{get_default_order(data_shape.size())};
    return {make_shared<op::Reshape>(data, input_order, output_data_shape)};
}

shared_ptr<Node> op::Squeeze::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Squeeze>(new_args.at(0), new_args.at(1));
}
