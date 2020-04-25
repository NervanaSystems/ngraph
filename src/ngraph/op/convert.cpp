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

#include <memory>

#include "ngraph/op/convert.hpp"
#include "ngraph/runtime/reference/convert.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Convert::type_info;

op::Convert::Convert(const Output<Node>& arg, const element::Type& destination_type)
    : Op({arg})
    , m_destination_type(destination_type)
{
    constructor_validate_and_infer_types();
}

void op::Convert::validate_and_infer_types()
{
    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

bool op::Convert::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

shared_ptr<Node> op::Convert::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Convert>(new_args.at(0), m_destination_type);
}

void op::Convert::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);

    adjoints.add_delta(x, make_shared<op::Convert>(delta, x.get_element_type()));
}

namespace
{
    template <element::Type_t ET>
    bool try_evaluate_convert(const EvaluatorTensorPtr& arg,
                              const EvaluatorTensorPtr& out,
                              size_t count)

    {
        return (ET == arg->get_element_type()) &&
               (runtime::reference::convert(arg->get_ptr<ET>(), out->get_ptr<ET>(), count), true);
    }

    bool
        evaluate_convert(const EvaluatorTensorPtr& arg, const EvaluatorTensorPtr& out, size_t count)
    {
        return try_evaluate_convert<element::Type_t::i8>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::i16>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::i32>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::i64>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::u8>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::u16>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::u32>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::u64>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::f32>(arg, out, count) ||
               try_evaluate_convert<element::Type_t::f64>(arg, out, count);
    }
}

bool op::v0::Convert::evaluate(const EvaluatorTensorVector& output_values,
                               const EvaluatorTensorVector& input_values)
{
    return evaluate_convert(
        input_values[0], output_values[0], output_values[0]->get_element_count());
}
