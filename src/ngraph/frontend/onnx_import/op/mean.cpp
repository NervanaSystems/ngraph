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

#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"

#include "mean.hpp"
#include "utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector mean(const Node& node)
            {
                auto sum = variadic::make_ng_variadic_op<ngraph::op::Add>(node).front();
                auto shape = sum->get_shape();

                // Create a Constant representing the number of inputs with the same shape as sum
                auto count = ngraph::op::Constant::create(
                    sum->get_element_type(),
                    shape,
                    std::vector<int>(shape_size(shape), node.get_ng_inputs().size()));

                return {sum / count};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
