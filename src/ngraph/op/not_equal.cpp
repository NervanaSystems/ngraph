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

#include "ngraph/op/not_equal.hpp"

using namespace std;
using namespace ngraph;

op::NotEqual::NotEqual(const NodeOutput& arg0, const NodeOutput& arg1)
    : BinaryElementwiseComparison("NotEqual", arg0, arg1)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::NotEqual::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<NotEqual>(new_source_outputs.at(0), new_source_outputs.at(1));
}
