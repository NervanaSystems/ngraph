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

#include "ngraph/ops/product.hpp"
#include "ngraph/function.hpp"
#include "ngraph/ops/broadcast.hpp"

using namespace std;
using namespace ngraph;

op::Product::Product(const std::shared_ptr<Node>& arg, const AxisSet& reduction_axes)
    : RequiresTensorViewArgs("Product", {arg})
    , m_reduction_axes(reduction_axes)
{
    auto& input = get_inputs().at(0);
    auto& input_element_type = input.get_element_type();
    if (input_element_type == element::boolean)
    {
        throw ngraph_error("Argument for product must have numeric element type");
    }

    auto input_shape = input.get_shape();

    for (auto axis : m_reduction_axes)
    {
        if (axis >= input_shape.size())
        {
            throw ngraph_error("Reduction axis for product is out of bounds");
        }
    }

    Shape result_shape;

    for (size_t i = 0; i < input_shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(input_shape.at(i));
        }
    }

    set_value_type_checked(input.get_element_type(), result_shape);
}

// TODO: Implement this.
void op::Product::generate_adjoints(autodiff::Adjoints& adjoints,
                                    const std::shared_ptr<Node>& delta)
{
    /*
    auto x = get_inputs().at(0).get_output().get_node();
    auto& x_shape = get_inputs().at(0).get_shape();

    adjoints.add_delta(x, make_shared<op::Broadcast>(delta, x_shape, m_reduction_axes));
*/
}
