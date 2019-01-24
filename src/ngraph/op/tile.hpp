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

#include <string>
#include <tuple>
#include <vector>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// An op directly representing PlaidML Tile code.
        ///
        /// N.B. Not all backends support Tile operations.
        class Tile;
    }
}

class ngraph::op::Tile final : public Node
{
public:
    Tile(const std::string& node_type,
         const std::string& function,
         const NodeVector& args,
         std::vector<std::tuple<element::Type, PartialShape>> outputs);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const final;

    const std::string& func() const { return m_function; }
private:
    std::string m_function;
    std::vector<std::tuple<element::Type, PartialShape>> m_output_shapes;
};
