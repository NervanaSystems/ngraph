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

#include <utility>

#include "ngraph/op/tile.hpp"

ngraph::op::Tile::Tile(const std::string& node_type,
                       const std::string& function,
                       const NodeVector& args,
                       std::vector<std::tuple<element::Type, PartialShape>> outputs)
    : Node{node_type, args, outputs.size()}
    , m_function{std::move(function)}
    , m_output_shapes{std::move(outputs)}
{
    constructor_validate_and_infer_types();
}

void ngraph::op::Tile::validate_and_infer_types()
{
    // TODO: It would be useful to have PlaidML deduce the output
    //       shapes, instead of having them passed in via the
    //       constructor and trusting that they're correct.
    //
    //       The primary barrier to doing so is that PlaidML
    //       placeholders always have a fixed number of dimensions but
    //       arbitrary dimension sizes, and the only way to pin them
    //       down to a concrete dimension size is to bind a tensor to
    //       them, which requires actually allocating the tensor.  In
    //       principal, we could fix this pretty easily; we'll need to
    //       know more about where the PlaidML API is going before
    //       doing so, though.
    //
    //       As a secondary barrier, we choose to always include a
    //       definition for the operation class, without linking to
    //       PlaidML, making it slightly tricker to access the PlaidML
    //       Tile analysis logic.

    std::size_t idx = 0;
    for (auto& output_shape : m_output_shapes)
    {
        set_output_type(idx++, std::get<0>(output_shape), std::get<1>(output_shape));
    }
}

std::shared_ptr<ngraph::Node> ngraph::op::Tile::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != get_input_size())
    {
        throw ngraph_error{"Tile node input counts cannot be changed for a given Tile function"};
    }
    return std::make_shared<Tile>(description(), m_function, new_args, m_output_shapes);
}
