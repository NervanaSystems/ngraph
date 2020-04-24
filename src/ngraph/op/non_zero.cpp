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

#include "ngraph/op/non_zero.hpp"
#include "ngraph/op/op.hpp"

using namespace ngraph;
using namespace std;

constexpr NodeTypeInfo op::v3::NonZero::type_info;

op::v3::NonZero::NonZero(const Output<Node>& arg)
    : Op({arg})
{
    constructor_validate_and_infer_types();
}

op::v3::NonZero::NonZero(const Output<Node>& arg, const std::string& output_type)
    : Op({arg})
    , m_output_type(EnumNames<element::Type_t>::as_enum(output_type))
{
    constructor_validate_and_infer_types();
}

op::v3::NonZero::NonZero(const Output<Node>& arg, const element::Type& output_type)
    : Op({arg})
    , m_output_type(output_type)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v3::NonZero::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v3::NonZero::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);
    const auto input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_integral() || input_et.is_real(),
                          "NonZero input data type needs to be a numeric type. Got: ",
                          input_et);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    set_output_type(0, m_output_type, PartialShape{input_shape.rank(), Dimension::dynamic()});
    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v3::NonZero::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v3::NonZero>(new_args.at(0), m_output_type);
}
