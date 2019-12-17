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

#include <cstddef>
#include <cstdint>
#include <memory>

#include "default_opset.hpp"
#include "expand.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector expand(const Node& node)
                {
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const std::shared_ptr<ngraph::Node> shape{node.get_ng_inputs().at(1)};

                    NGRAPH_CHECK(shape->is_constant(),
                                 "Ngraph does not support dynamic braodcasting for Expand op.");

                    std::vector<std::size_t> shape_vector =
                        ngraph::as_type_ptr<default_opset::Constant>(shape)
                            ->get_vector<std::size_t>();

                    const ngraph::Shape shape_shape{shape_vector};
                    return {ngraph::op::numpy_style_broadcast(data, shape_shape)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
