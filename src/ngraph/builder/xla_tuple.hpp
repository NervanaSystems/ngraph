// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <vector>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace builder
    {
        class Tuple : public Node
        {
        public:
            Tuple(const std::vector<std::shared_ptr<Node>>& nodes);

            std::shared_ptr<Node> get_tuple_element(size_t i);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override;

        protected:
            Nodes m_nodes;
        };

        std::shared_ptr<Node> get_tuple_element(std::shared_ptr<Node> tuple, size_t i);
    }
}
