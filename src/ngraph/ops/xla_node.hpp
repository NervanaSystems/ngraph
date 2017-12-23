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

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        class XLATuple;

        class XLANode : public Node
        {
        protected:
            XLANode(const std::string& node_type,
                    const std::vector<std::shared_ptr<Node>>& arguments)
                : Node(node_type, arguments)
            {
            }

        public:
            virtual std::shared_ptr<const XLATuple> get_tuple_value() const = 0;
            virtual const Nodes& get_tuple_elements() const = 0;
        };
    }
}
