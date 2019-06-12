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

#include "ngraph/runtime/plaidml/plaidml_ops_implicit_broadcast.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace vp = vertexai::plaidml;

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            NGRAPH_PLAIDML_OP_CLASS(ImplImplicitBroadcast, OpImpl<plaidml::op::ImplicitBroadcast>);
        }
    }
}

ngraph::runtime::plaidml::op::ImplicitBroadcast::ImplicitBroadcast(std::shared_ptr<Node> input,
                                                                   const Shape& shape)
    : Op{"ImplicitBroadcast", {input}}
    , m_shape{shape}
{
    constructor_validate_and_infer_types();
}

void ngraph::runtime::plaidml::op::ImplicitBroadcast::validate_and_infer_types()
{
    set_output_type(0, input(0).get_element_type(), m_shape);
}

std::shared_ptr<ngraph::Node> ngraph::runtime::plaidml::op::ImplicitBroadcast::copy_with_new_args(
    const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error{"Implicit broadcast requires a single input"};
    }
    return std::make_shared<ImplicitBroadcast>(new_args.at(0), m_shape);
}

void ngraph::runtime::plaidml::ImplImplicitBroadcast::Apply()
{
    check_inputs(1);
    check_outputs(1);
    set_output(op_input(0));
}
