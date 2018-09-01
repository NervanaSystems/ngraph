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

#include <sstream>

#include "ngraph/op/parameter.hpp"

using namespace std;
using namespace ngraph;

op::Parameter::Parameter(const element::Type& element_type,
                         const Shape& shape,
                         const bool cacheable)
    : Op("Parameter", {})
    , m_cacheable(cacheable)
    , m_shape(shape)
    , m_element_type(element_type)
{
    constructor_validate_and_infer_types();
}

void op::Parameter::validate_and_infer_types()
{
    Op::validate_and_infer_types();
    set_output_type(0, m_element_type, m_shape);
}

shared_ptr<Node> op::Parameter::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Parameter>(m_element_type, m_shape);
}

void op::Parameter::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);
}
