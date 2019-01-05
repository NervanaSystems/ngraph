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

#include <utility>

#include "ngraph/runtime/plaidml/plaidml_impl.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_tile.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplTile, OpImpl<op::Tile>);
        }
    }
}

ngraph::runtime::plaidml::op::Tile::Tile(
    const std::string& node_type,
    vertexai::plaidml::function function,
    const NodeVector& args,
    std::vector<std::tuple<element::Type, PartialShape>> outputs)
    : Node{node_type, args, outputs.size()}
    , m_function{std::move(function)}
    , m_output_shapes{std::move(outputs)}
{
    constructor_validate_and_infer_types();
}

void ngraph::runtime::plaidml::op::Tile::validate_and_infer_types()
{
    // TODO: It would be useful to have PlaidML deduce the output
    //       shapes, instead of having them passed in via the
    //       constructor.  The primary barrier to doing so is that
    //       PlaidML placeholders always have a fixed number of
    //       dimensions but arbitrary dimension sizes, and the only way
    //       to pin them down to a concrete dimension size is to bind a
    //       tensor to them, which requires actually allocating the
    //       tensor.  In principal, we could fix this pretty easily;
    //       we'll need to know more about where the PlaidML API is
    //       going before doing so, though.
    if (get_input_size() != m_function.num_inputs())
    {
        throw ngraph_error{"Incorrect input count for Tile operation node"};
    }

    if (m_output_shapes.size() != m_function.num_outputs())
    {
        throw ngraph_error{"Incorrect output count for Tile operation node"};
    }

    std::size_t idx = 0;
    for (auto& output_shape : m_output_shapes)
    {
        set_output_type(idx++, std::get<0>(output_shape), std::get<1>(output_shape));
    }
}

std::shared_ptr<ngraph::Node>
    ngraph::runtime::plaidml::op::Tile::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != get_input_size())
    {
        throw ngraph_error{"Tile node input counts cannot be changed for a given Tile function"};
    }
    return std::make_shared<Tile>(description(), m_function, new_args, m_output_shapes);
}

void ngraph::runtime::plaidml::ImplTile::Apply()
{
    vertexai::plaidml::function::positional_t inputs;

    for (std::size_t idx = 0; idx < op().get_input_size(); ++idx)
    {
        inputs.emplace_back(op_input(idx));
    }

    auto app = op().func().apply(inputs);

    for (std::size_t idx = 0; idx < op().get_output_size(); ++idx)
    {
        set_output(idx, app.get_output(idx));
    }
}
