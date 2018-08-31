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

#include <algorithm>
#include <sstream>

#include "ngraph/function.hpp"
#include "ngraph/op/reverse.hpp"

using namespace std;
using namespace ngraph;

op::Reverse::Reverse(const shared_ptr<Node>& arg, const AxisSet& reversed_axes)
    : Op("Reverse", check_single_output_args({arg}))
    , m_reversed_axes(reversed_axes)
{
    constructor_validate_and_infer_types();
}

void op::Reverse::validate_and_infer_types()
{
    auto input_shape = get_input_shape(0);
    auto input_rank = input_shape.size();

    // Make sure all reversed axis indices are valid.
    for (size_t axis : m_reversed_axes)
    {
        NODE_VALIDATION_ASSERT(this, axis < input_rank)
            << "Reverse axis (" << axis << ") is out of bounds (argument shape: " << input_shape
            << ").";
    }

    set_output_type(0, get_input_element_type(0), input_shape);
}

shared_ptr<Node> op::Reverse::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Reverse>(new_args.at(0), m_reversed_axes);
}

void op::Reverse::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);

    adjoints.add_delta(x, make_shared<op::Reverse>(delta, m_reversed_axes));
}
