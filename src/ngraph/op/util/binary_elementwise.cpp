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

#include <memory>

#include "ngraph/log.hpp"
#include "ngraph/op/util/binary_elementwise.hpp"

using namespace std;
using namespace ngraph;

op::util::BinaryElementwise::BinaryElementwise(const std::string& node_type,
                                               const std::shared_ptr<Node>& arg0,
                                               const std::shared_ptr<Node>& arg1)
    : Op(node_type, check_single_output_args(NodeVector{arg0, arg1}))
{
}

void op::util::BinaryElementwise::validate_and_infer_types()
{
    validate_and_infer_elementwise();
}
