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

#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

op::OneHot::OneHot(const shared_ptr<Node>& arg, const Shape& shape, size_t one_hot_axis)
    : Op("OneHot", check_single_output_args({arg}))
    , m_shape(shape)
    , m_one_hot_axis(one_hot_axis)
{
    constructor_validate_and_infer_types();

    auto& input = m_inputs.at(0);
    auto& input_element_type = input.get_element_type();

    NODE_VALIDATION_ASSERT(this, one_hot_axis < shape.size())
        << "One-hot axis (" << one_hot_axis
        << ") is out of bounds (requested result shape: " << shape << ").";

    auto expected_input_shape = shape;
    expected_input_shape.erase(expected_input_shape.begin() + one_hot_axis);

    NODE_VALIDATION_ASSERT(this, input.get_shape() == expected_input_shape)
        << "Argument shape " << input.get_shape() << " does not match the expected shape of "
        << expected_input_shape << ".";

    set_output_type(0, input_element_type, shape);
}

shared_ptr<Node> op::OneHot::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<OneHot>(new_args.at(0), m_shape, m_one_hot_axis);
}
