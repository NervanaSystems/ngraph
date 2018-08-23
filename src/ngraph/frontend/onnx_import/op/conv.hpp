/*******************************************************************************
 * Copyright 2018 Intel Corporation
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

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/strides.hpp"

#include "ngraph/frontend/onnx_import/exceptions.hpp"
#include "ngraph/frontend/onnx_import/utils/broadcasting.hpp"
#include "ngraph/frontend/onnx_import/utils/convpool.hpp"
#include "ngraph/frontend/onnx_import/wrappers/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace detail
            {
                std::shared_ptr<ngraph::op::Op>
                    make_ng_convolution(const std::shared_ptr<ngraph::Node>& data,
                                        const std::shared_ptr<ngraph::Node>& filters,
                                        const ngraph::Strides& strides,
                                        const ngraph::Strides& dilations,
                                        const ngraph::CoordinateDiff& padding_below,
                                        const ngraph::CoordinateDiff& padding_above,
                                        int groups);
            }

            /**
             * @brief Performs ONNX Conv operation.
             *
             * @param node   The ONNX node object representing this operation.
             *
             * @return The vector containing Ngraph nodes producing output of ONNX convolution
             *         operation.
             */
            NodeVector conv(const Node& node);

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
