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
    : RequiresTensorViewArgs("OneHot", {arg})
    , m_shape(shape)
    , m_one_hot_axis(one_hot_axis)
{
    auto& input = m_inputs.at(0);
    auto& input_element_type = input.get_element_type();

    if (one_hot_axis >= shape.size())
    {
        throw ngraph_error("One-hot axis is out of bounds");
    }

    auto expected_input_shape = shape;
    expected_input_shape.erase(expected_input_shape.begin() + one_hot_axis);

    if (input.get_shape() != expected_input_shape)
    {
        throw ngraph_error("One-hot argument shape is not compatible with desired output shape");
    }

    set_value_type_checked(make_shared<TensorViewType>(input_element_type, shape));
}

shared_ptr<Node> op::OneHot::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<OneHot>(new_args.at(0), m_shape, m_one_hot_axis);
}
