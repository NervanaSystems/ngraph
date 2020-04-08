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

#pragma once

#include <memory>

#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        /// \brief  The FusedOpDecomposition pass is used to decompose a fused op
        /// if it is not supported by the backend.
        ///
        /// \details The pass is run on each node in the computation graph. If the
        /// node is a fused op and is not supported by the backend, the pass
        /// decomposes it and is run recusrsively until no more fused ops
        /// can be found or the new ops are supported by the backend.
        /// It also adds provenance tags along the way to each op for easy reference
        /// and debugging.
        /// The backend can optionally provide a callback function to indicate if a
        /// fused op is supported or not while registering the pass.
        ///
        /// In the example shown below, the original graph has a fused Gelu op.
        /// After applying this pass, the Gelu op is decomposed into new ops which together
        /// perform the same operation as Gelu.
        /// <table>
        /// <tr><th>Before the pass:</th>
        ///      <th> After the pass</th>
        /// </tr>
        /// <tr>
        ///      <td> \image html decompose_gelu_pre.svg </td>
        ///      <td> \image html decompose_gelu_post.svg </td>
        /// </tr>
        /// </table>
        class NGRAPH_API FusedOpDecomposition : public NodePass
        {
        public:
            /// \brief  Function signature type for callback used to check whether provided node
            ///         is supported by backend.
            using op_query_t = std::function<bool(const Node& node)>;

            ///
            /// \brief      Constructor for the Fused operation decomposition pass.
            ///
            /// \param[in]  callback  The function object used to determine whether current backend
            ///                       provide direct support for passed node. Should have signature:
            ///                       bool fn(const Node&)
            ///
            FusedOpDecomposition(op_query_t callback = nullptr);
            bool run_on_node(std::shared_ptr<ngraph::Node> node) override;

        private:
            /// \brief A function returning whether provided Node is supported by current backend.
            ///        The returned bool value is used to control whether decompose operator or not.
            op_query_t m_has_direct_support = nullptr;
        };
    }
}
