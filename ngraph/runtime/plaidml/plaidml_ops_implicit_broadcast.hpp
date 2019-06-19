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

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace op
            {
                // Implements NumPy-style broadcast semantics by passing its single argument through to its
                // output and pretending that this changes the shape.  The creator of this node is responsible
                // for ensuring that the downstream operation will perform a NumPy-style broadcast.
                class ImplicitBroadcast;
            }
        }
    }
}

class ngraph::runtime::plaidml::op::ImplicitBroadcast final : public ngraph::op::Op
{
public:
    ImplicitBroadcast(std::shared_ptr<Node> input, const Shape& shape);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const final;

private:
    Shape m_shape;
};
