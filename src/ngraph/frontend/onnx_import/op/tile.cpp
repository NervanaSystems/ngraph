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

#include <memory>

#include "core/node.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/experimental/tile.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector tile(const Node& node)
                {
                    auto input = node.get_ng_inputs().at(0);
                    auto repeats = node.get_ng_inputs().at(1);

                    repeats = std::make_shared<ngraph::op::Convert>(repeats, element::i64);

                    return {std::make_shared<ngraph::op::Tile>(input, repeats)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
