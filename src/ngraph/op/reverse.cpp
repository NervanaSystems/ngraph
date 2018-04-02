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

#include <algorithm>
#include <sstream>

#include "ngraph/function.hpp"
#include "ngraph/op/reverse.hpp"

using namespace std;
using namespace ngraph;

op::Reverse::Reverse(const shared_ptr<Node>& arg, const AxisSet& reversed_axes)
    : RequiresTensorViewArgs("Reverse", {arg})
    , m_reversed_axes(reversed_axes)
{
    auto& input = get_inputs().at(0);
    auto input_shape = input.get_shape();
    auto input_rank = input_shape.size();

    // Make sure all reversed axis indices are valid.
    for (size_t axis : reversed_axes)
    {
        if (axis >= input_rank)
        {
            stringstream ss;
            ss << "Reverse axis " << axis << " is out of bounds (input rank is " << input_rank
               << ").";
            throw ngraph_error(ss.str());
        }
    }

    set_value_type_checked(input.get_element_type(), input_shape);
}

shared_ptr<Node> op::Reverse::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Reverse>(new_args.at(0), m_reversed_axes);
}

void op::Reverse::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_input_op(0);

    adjoints.add_delta(x, make_shared<op::Reverse>(delta, m_reversed_axes));
}
