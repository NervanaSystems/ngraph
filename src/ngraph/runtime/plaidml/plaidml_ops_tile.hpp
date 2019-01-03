//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <tuple>
#include <vector>

#include <plaidml/plaidml++.h>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace op
            {
                /// An op directly representing PlaidML Tile code.
                class Tile;
            }
        }
    }
}

class ngraph::runtime::plaidml::op::Tile final : public Node
{
public:
    Tile(const std::string& node_type,
         vertexai::plaidml::function function,
         const NodeVector& args,
         std::vector<std::tuple<element::Type, PartialShape>> outputs);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const final;

    vertexai::plaidml::function func() const { return m_function; }
private:
    vertexai::plaidml::function m_function;
    std::vector<std::tuple<element::Type, PartialShape>> m_output_shapes;
};
