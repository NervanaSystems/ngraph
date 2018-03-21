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

#include <cassert>
#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "backprop_function.hpp"
#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;

std::shared_ptr<Function> autodiff::backprop_function(const std::shared_ptr<Function>& f)
{
    auto Y_out = f->get_output_op(0);
    auto Xs = f->get_parameters();
    auto C = std::make_shared<op::Parameter>(Y_out->get_element_type(), Y_out->get_shape());
    std::vector<std::shared_ptr<Node>> dYdXs(Xs.size());
    transform(Xs.begin(), Xs.end(), dYdXs.begin(), [C, Y_out](const std::shared_ptr<Node>& X) {
        return Y_out->backprop_node(X, C);
    });
    std::vector<std::shared_ptr<op::Parameter>> params(Xs);
    params.push_back(C);
    return std::make_shared<Function>(dYdXs, params);
}
