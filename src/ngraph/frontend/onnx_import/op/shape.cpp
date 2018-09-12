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

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

#include "ngraph/op/constant.hpp"

#include "shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector shape(const Node& node)
            {
                auto data = node.get_ng_inputs().at(0);
                auto data_shape = data->get_shape();

                return {std::make_shared<ngraph::op::Constant>(
                    ngraph::element::i64, Shape{data_shape.size()}, data_shape)};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
