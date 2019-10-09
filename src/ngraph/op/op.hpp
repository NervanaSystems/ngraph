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

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace op
    {
        /// Root of all actual ops
        class Op : public Node
        {
        public:
            virtual bool is_op() const override { return true; }
        protected:
            Op()
                : Node()
            {
            }
            Op(const NodeVector& arguments);
            Op(const OutputVector& arguments);
            Op(const std::string& node_type, const NodeVector& arguments);
        };
    }
}
