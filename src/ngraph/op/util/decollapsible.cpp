/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <memory>
#include <sstream>

#include "ngraph/except.hpp"
#include "ngraph/op/util/decollapsible.hpp"
#include "ngraph/type/type.hpp"

using namespace ngraph;
using namespace std;

op::util::Decollapsible::Decollapsible(const std::string& node_type,
                                       const NodeVector& args,
                                       std::shared_ptr<Node> original_node)
    : RequiresTensorViewArgs(node_type, args)
    , m_original_node(original_node)
{
}
