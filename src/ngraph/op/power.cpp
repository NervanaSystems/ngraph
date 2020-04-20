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

#include "ngraph/op/power.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/runtime/reference/power.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ v0 -------------------------------------------

constexpr NodeTypeInfo op::v0::Power::type_info;

op::v0::Power::Power(const Output<Node>& arg0,
                     const Output<Node>& arg1,
                     const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Power::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Power>(new_args.at(0), new_args.at(1), this->get_autob());
}

void op::v0::Power::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);

    auto log_x = make_shared<op::Log>(x);

    adjoints.add_delta(x, delta * y * shared_from_this() / x);
    adjoints.add_delta(y, delta * shared_from_this() * log_x);
}

namespace
{
    template <element::Type_t ET>
    bool try_evaluate_power(const EvaluatorTensorPtr& arg0,
                            const EvaluatorTensorPtr& arg1,
                            const EvaluatorTensorPtr& out,
                            const op::AutoBroadcastSpec& broadcast_spec)
    {
        return (ET == arg0->get_element_type()) && (runtime::reference::power(arg0->get_ptr<ET>(),
                                                                              arg1->get_ptr<ET>(),
                                                                              out->get_ptr<ET>(),
                                                                              arg0->get_shape(),
                                                                              arg1->get_shape(),
                                                                              broadcast_spec),
                                                    true);
    }

    bool evaluate_power(const EvaluatorTensorPtr& arg0,
                        const EvaluatorTensorPtr& arg1,
                        const EvaluatorTensorPtr& out,
                        const op::AutoBroadcastSpec& broadcast_spec)
    {
        return try_evaluate_power<element::Type_t::i8>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::i16>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::i32>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::i64>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::u8>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::u16>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::u32>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::u64>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::f32>(arg0, arg1, out, broadcast_spec) ||
               try_evaluate_power<element::Type_t::f64>(arg0, arg1, out, broadcast_spec);
    }
}

bool op::v0::Power::evaluate(const EvaluatorTensorVector& outputs,
                             const EvaluatorTensorVector& inputs)
{
    return evaluate_power(inputs[0], inputs[1], outputs[0], get_autob());
}

// ------------------------------ v1 -------------------------------------------

constexpr NodeTypeInfo op::v1::Power::type_info;

op::v1::Power::Power(const Output<Node>& arg0,
                     const Output<Node>& arg1,
                     const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Power::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Power>(new_args.at(0), new_args.at(1), this->get_autob());
}

void op::v1::Power::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);

    auto log_x = make_shared<op::Log>(x);

    adjoints.add_delta(x, delta * y * shared_from_this() / x);
    adjoints.add_delta(y, delta * shared_from_this() * log_x);
}

bool op::v1::Power::evaluate(const EvaluatorTensorVector& outputs,
                             const EvaluatorTensorVector& inputs)
{
    return evaluate_power(inputs[0], inputs[1], outputs[0], get_autob());
}
