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

#include <algorithm>
#include <memory>
#include <sstream>

#include "ngraph/except.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/types/type.hpp"

using namespace ngraph;
using namespace std;

op::RequiresTensorViewArgs::RequiresTensorViewArgs(const std::string& node_type,
                                                   const std::vector<std::shared_ptr<Node>>& args)
    : Node(node_type, args)
{
    for (auto arg : args)
    {
        if (nullptr == std::dynamic_pointer_cast<const TensorViewType>(arg->get_value_type()))
        {
            throw ngraph_error("Arguments for node type \"" + node_type +
                               "\" must be tensor views");
        }
    }
}
