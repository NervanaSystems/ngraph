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

#include <memory>

#include "ngraph/op/convert.hpp"

using namespace std;
using namespace ngraph;

op::Convert::Convert(const NodeOutput& arg, const element::Type& element_type)
    : Op("Convert", {arg})
    , m_element_type(element_type)
{
    constructor_validate_and_infer_types();
}

void op::Convert::validate_and_infer_types()
{
    set_output_type(0, m_element_type, get_input_partial_shape(0));
}

shared_ptr<Node>
    op::Convert::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<Convert>(new_source_outputs.at(0), m_element_type);
}

void op::Convert::build_backprop(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_input_source_output(0);

    adjoints.add_output_delta(x, make_shared<op::Convert>(delta, x.get_element_type()));
}
