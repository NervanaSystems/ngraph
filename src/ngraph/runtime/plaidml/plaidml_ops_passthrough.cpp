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

#include <utility>

#include "ngraph/except.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace vp = vertexai::plaidml;

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplPassthrough, OpImpl<op::Passthrough>);
        }
    }
}

void ngraph::runtime::plaidml::ImplPassthrough::Apply()
{
    if (op().language() != "Tile")
    {
        throw unsupported_op{"Unsupported operation language: " + op().language()};
    }

    vertexai::plaidml::function::positional_t inputs;

    for (std::size_t idx = 0; idx < op().get_input_size(); ++idx)
    {
        inputs.emplace_back(op_input(idx));
    }

    auto app = vp::function{op().function()}.apply(inputs);

    for (std::size_t idx = 0; idx < op().get_output_size(); ++idx)
    {
        set_output(idx, app.get_output(idx));
    }
}
