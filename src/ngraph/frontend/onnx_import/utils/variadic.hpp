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

#pragma once

#include <numeric>

#include "core/node.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace variadic
        {
            /// \brief Create an nGraph version of an ONNX variadic operation.
            ///        This creates a subgraph with a series of binary operations.
            ///
            /// \param node Incoming ONNX opearation.
            ///
            /// \tparam T   Class of an nGraph binary operation (e.g. Add, Minimum, Maximum)
            ///
            /// \return nGraph node equivalent of the ONNX operation
            template <class T>
            inline NodeVector make_ng_variadic_op(const Node& node)
            {
                NodeVector ng_inputs{node.get_ng_inputs()};

                // Templated binary operation - Creates Add, Minimum, Maximum, etc.
                auto binary_operation = [](const std::shared_ptr<ngraph::Node>& arg0,
                                           const std::shared_ptr<ngraph::Node>& arg1) {
                    return std::make_shared<T>(arg0, arg1);
                };

                // Create a result node as a series of binary operations
                auto result = std::accumulate(
                    std::next(std::begin(ng_inputs)), // First operand value - the second input
                    std::end(ng_inputs),              // Last value - final input
                    ng_inputs.front(),                // Initial value - first input
                    binary_operation);

                return {result};
            }

        } // namespace variadic

    } // namespace  onnx_import

} // namespace  ngraph
