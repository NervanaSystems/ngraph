//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

using namespace ngraph;

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic()
    : Op()
{
}

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const Output<Node>& arg)
    : Op({arg})
{
}

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const std::shared_ptr<Node>& arg)
    : Op(check_single_output_args({arg}))
{
}

op::util::UnaryElementwiseArithmetic::UnaryElementwiseArithmetic(const std::string& node_type,
                                                                 const std::shared_ptr<Node>& arg)
    : Op(node_type, check_single_output_args({arg}))
{
}

void op::util::UnaryElementwiseArithmetic::validate_and_infer_types()
{
    validate_and_infer_elementwise_arithmetic();
}

bool op::util::UnaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}
