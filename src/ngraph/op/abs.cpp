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

#include "ngraph/op/abs.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sign.hpp"

void ngraph::op::Abs::generate_adjoints(autodiff::Adjoints& adjoints,
                                        const std::shared_ptr<Node>& delta)
{
    auto x = get_inputs().at(0).get_output().get_node();

    adjoints.add_delta(x, delta * std::make_shared<op::Sign>(x));
}
