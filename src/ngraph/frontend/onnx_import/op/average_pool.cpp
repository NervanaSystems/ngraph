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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/strides.hpp"

#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            NodeVector average_pool(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};
                std::shared_ptr<ngraph::Node>& data{ng_inputs.at(0)};

                // We assume data are in [D1,...,DN] format thus we subtract [N,C] dimensions.
                // std::size_t spatial_dims = data->get_shape().size() - 2;  // get spatial dimensions

                Shape kernel_shape = attribute::get_kernel_shape(node);

                auto strides{attribute::get_strides(node)};
                auto dilations{attribute::get_dilations(node)};

                auto paddings{attribute::get_pads(node)};
                const auto& padding_below{paddings.first};
                const auto& padding_above{paddings.second};

                Shape padding_below_shape{std::begin(padding_below), std::end(padding_below)};
                Shape padding_above_shape{std::begin(padding_above), std::end(padding_above)};
                bool include_padding_in_avg_computation = false;

                return {std::make_shared<ngraph::op::AvgPool>(data,
                                                              kernel_shape,
                                                              strides,
                                                              padding_below_shape,
                                                              padding_above_shape,
                                                              include_padding_in_avg_computation)};
            }

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
