//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector shape(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto data_ps = data->get_input_partial_shape(0);

                    if (data_ps.is_static())
                    {
                        const auto data_shape = data_ps.to_shape();

                        return {default_opset::Constant::create(
                            ngraph::element::i64, Shape{data_shape.size()}, data_shape)};
                    }
                    else
                    {
                        return {std::make_shared<default_opset::ShapeOf>(data)};
                    }
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
