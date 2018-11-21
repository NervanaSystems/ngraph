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

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        /// \brief Builds a subgraph broadcasting `scalar_node`, which must produce a scalar, to
        ///        the shape returned by `shape_node`, which must return a shape (vector of u64).
        std::shared_ptr<Node> broadcast_scalar_to(std::shared_ptr<Node> shape_node, std::shared_ptr<Node> scalar_node);
    } // namespace builder
} // namespace ngraph
