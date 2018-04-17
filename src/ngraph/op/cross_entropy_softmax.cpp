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

#include "ngraph/op/cross_entropy_softmax.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

op::CrossEntropySoftMax::CrossEntropySoftMax(std::shared_ptr<ngraph::Node> y,
                                             std::shared_ptr<ngraph::Node> t,
                                             std::shared_ptr<Node> old_cross_entropy,
                                             size_t axis)
    : Decollapsible("CrossEntropySoftMax", {y, t}, old_cross_entropy)
    , m_original_cross_entropy(old_cross_entropy)
{
    if (y->get_shape() != t->get_shape())
    {
        throw ngraph_error("shapes of y and t are different");
    }

    if (y->get_element_type() != t->get_element_type())
    {
        throw ngraph_error("element types of y and t are different");
    }

    if (y->get_shape().size() != 2)
    {
        throw ngraph_error("number of dimensions isn't equal to 2");
    }

    //double check if we are only reducing across each sample in a batch
    //and not across the entire batch
    set_value_type_checked(y->get_element_type(), Shape{y->get_shape().at(1 - axis)});
}

void op::CrossEntropySoftMax::generate_adjoints(autodiff::Adjoints& adjoints,
                                                const NodeVector& deltas)
{
    adjoints.add_delta(get_input_op(0), get_input_op(0) - get_input_op(1));
}
