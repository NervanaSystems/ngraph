//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "cum_sum.hpp"
#include "ngraph/op/cum_sum.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector cum_sum(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    auto exclusive = node.get_attribute_value<int64_t>("exclusive", 0);
                    auto reverse = node.get_attribute_value<int64_t>("reverse", 0);
                    
                    if (node.get_ng_inputs().size() > 1)
                    {
                        auto axis = node.get_ng_inputs().at(1);  // optional input, 0-D tensor
                        return NodeVector{std::make_shared<ngraph::op::CumSum>(data, axis, exclusive, reverse)};
                    }
                    
                    return NodeVector{std::make_shared<ngraph::op::CumSum>(data, exclusive=exclusive, reverse=reverse)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
