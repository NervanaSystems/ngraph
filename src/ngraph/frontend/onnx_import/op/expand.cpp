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

#include "expand.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_8
            {
                std::shared_ptr<ngraph::Node> get_rank_from_shape(
                    std::shared_ptr<ngraph::Node> shapeof) {
                        auto shapeof_shape = std::make_shared<ngraph::op::ShapeOf>(shapeof);
                        return std::make_shared<ngraph::op::Reshape>(
                            shapeof_shape, ngraph::AxisVector{0}, ngraph::Shape{});
                }
                std::shared_ptr<ngraph::Node> ng_range(
                    std::shared_ptr<ngraph::Node> rank_scalar) {
                    return std::make_shared<ngraph::op::Range>(
                        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0}),
                        rank_scalar,
                        ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1}));
                }       
                std::shared_ptr<ngraph::Node> get_range_from_shape(
                    std::shared_ptr<ngraph::Node> shapeof) {
                return ng_range(get_rank_from_shape(shapeof));
                }

                NodeVector expand(const Node& node)
                {
                    const std::shared_ptr<ngraph::Node> data{node.get_ng_inputs().at(0)};
                    const std::shared_ptr<ngraph::Node> shape{node.get_ng_inputs().at(1)};
                    //std::vector<int64_t> shape_in{1};
                    auto out_shapeOf = std::make_shared<ngraph::op::ShapeOf>(data);

                    auto scalar = std::make_shared<ngraph::op::Reshape>(
                                    data, ngraph::AxisVector{0}, ngraph::Shape{});

                   //auto constant_shape = std::make_shared<ngraph::op::Constant>(element::i64, ngrap::Shape{}, shape_in);
                    //auto data_scalar = std::make_shared<ngraph::op::Constant>(element::i64, ngraph::Shape{}, ngraph::AxisSet{});
                    //auto data_reshape = std::make_shared<ngraph::op::DynReshape>(data, data_scalar);
                    auto out_bcast = std::make_shared<ngraph::op::DynBroadcast>(scalar, out_shapeOf, get_range_from_shape(out_shapeOf));
                    return {
                        out_bcast
                        //std::make_shared<ngraph::op::DynBroadcast>(data, shape, constant_shape)
                        };

                }

            } // namespace set_8

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
