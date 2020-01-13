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

#include "ngraph/frontend/fluid/operators/layout_converter.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph::fluid;

constexpr NodeTypeInfo LayoutConverter::type_info;

LayoutConverter::LayoutConverter(const Output<Node>& x, const int mode)
    : FusedOp({x})
    , m_mode(mode)
{
    constructor_validate_and_infer_types();
}

NodeVector LayoutConverter::decompose_op() const
{
    auto x = input_value(0);
    auto x_shape = get_input_shape(0);
    int mode = get_mode();

    NODE_VALIDATION_CHECK(this, x_shape.size() == 4, "Input rank is not 4");

    AxisVector axis_vec;

    switch (mode)
    {
    case 1: axis_vec = {0, 3, 1, 2}; break;
    case 2: axis_vec = {0, 2, 3, 1}; break;
    default: throw ngraph_error("Unsupported layout convert mode");
    }

    Shape out_shape = x_shape;

    for (size_t i = 0; i < 4; ++i)
    {
        out_shape[i] = x_shape[axis_vec[i]];
    }

    return {make_shared<op::Reshape>(x, axis_vec, out_shape)};
}

shared_ptr<Node> LayoutConverter::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);

    return make_shared<LayoutConverter>(new_args.at(0), get_mode());
}

void LayoutConverter::pre_validate_and_infer_types()
{
    auto shape = get_input_partial_shape(0);

    if (shape.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}
