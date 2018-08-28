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

#include <algorithm>
#include <memory>
#include <sstream>

#include "ngraph/except.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;
using namespace std;

op::util::RequiresTensorViewArgs::RequiresTensorViewArgs(const std::string& node_type,
                                                         const NodeVector& args)
    : Op(node_type, args)
{
    for (auto arg : args)
    {
        if (arg->get_output_size() != 1)
        {
            throw ngraph_error("Arguments for node type \"" + node_type +
                               "\" must be tensor views");
        }
    }
}
