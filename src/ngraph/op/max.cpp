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

#include "ngraph/op/max.hpp"

using namespace std;
using namespace ngraph;

op::Max::Max(const NodeOutput& arg, const AxisSet& reduction_axes)
    : ArithmeticReduction("Max", arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Max::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<Max>(new_source_outputs.at(0), m_reduction_axes);
}
