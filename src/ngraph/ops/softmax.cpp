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

#include "ngraph/ops/softmax.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"

void ngraph::op::Softmax::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const std::shared_ptr<Node>& delta)
{
    auto x = get_input_op(0);
    auto z = delta * shared_from_this();

    AxisSet axes;
    for (size_t i = 0; i < z->get_shape().size(); i++)
    {
        axes.insert(i);
    }
    auto zs = std::make_shared<op::Sum>(z, axes);

    auto dot = std::make_shared<op::Dot>(shared_from_this(), zs);
    adjoints.add_delta(x, z - dot);
}
