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

#pragma once

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        /**
         * @brief Permute axes according to specified axes_order parameter.
         *
         * @param node The node which axes we want to permute.
         * @param axes_order The permutation of node tensor axes.
         *
         * @return: New node with permuted axes.
         */
        std::shared_ptr<ngraph::Node> reorder_axes(const std::shared_ptr<ngraph::Node>& node,
                                                   std::vector<int> axes_order);

        /**
         * @brief Return transposed tensor (with axes in reversed order).
         *
         * @param node Input tensor we want to transpose
         *
         * @return: New node with reversed dimensions.
         */
        std::shared_ptr<ngraph::Node> transpose(const std::shared_ptr<ngraph::Node>& node);
    } // namespace onnx_import

} // namespace ngraph
