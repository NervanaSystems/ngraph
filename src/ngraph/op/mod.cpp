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
#include "ngraph/op/mod.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/runtime/reference/mod.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::Mod::type_info;

op::v1::Mod::Mod(const Output<Node>& A,
                 const Output<Node>& B,
                 const AutoBroadcastSpec& auto_broadcast)
    : FusedOp({A, B})
    , m_auto_broadcast(auto_broadcast)
{
}

bool ngraph::op::v1::Mod::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

OutputVector op::v1::Mod::decompose_op() const
{
    const auto dividend = make_shared<op::v0::Abs>(input_value(0));
    const auto dividend_sign = make_shared<op::v0::Sign>(input_value(0));
    const auto dividend_et = dividend->get_output_element_type(0);
    const auto divisor = make_shared<op::v0::Abs>(input_value(1));

    // truncated(a / b)
    auto division = make_shared<op::v0::Convert>(
        make_shared<op::v1::Divide>(dividend, divisor, m_auto_broadcast), ngraph::element::i64);
    division = make_shared<op::v0::Convert>(division, dividend_et);
    // truncated(a / b) * b
    const auto multiplication = make_shared<op::v1::Multiply>(division, divisor, m_auto_broadcast);
    // a mod b = a - truncated(a / b) * b
    const auto mod = make_shared<op::v1::Subtract>(dividend, multiplication, m_auto_broadcast);

    // apply sign of dividend
    return {make_shared<op::v1::Multiply>(dividend_sign, mod, m_auto_broadcast)};
}

shared_ptr<Node> op::v1::Mod::clone_with_new_inputs(const OutputVector& new_args) const
{
    return make_shared<Mod>(new_args.at(0), new_args.at(1), m_auto_broadcast);
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& output,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        using T = typename element_type_traits<ET>::value_type;

        Shape arg0_shape = arg0->get_shape();
        Shape arg1_shape = arg1->get_shape();

        Shape output_shape;
        if (arg0_shape == arg1_shape)
        {
            output_shape = arg0_shape;
        }
        else if (broadcast_spec == op::AutoBroadcastType::NONE)
        {
            return false;
        }
        else
        {
            const auto& broadcast_shapes =
                builder::get_numpy_broadcast_shapes({arg0_shape, arg1_shape});
            output_shape = broadcast_shapes.first;
        }

        output->set_element_type(arg0->get_element_type());
        output->set_shape(output_shape);

        runtime::reference::mod<T>(arg0->get_data_ptr<ET>(),
                                   arg1->get_data_ptr<ET>(),
                                   output->get_data_ptr<ET>(),
                                   arg0_shape,
                                   arg1_shape,
                                   broadcast_spec);
        return true;
    }

    bool evaluate_mod(const HostTensorPtr& arg0,
                      const HostTensorPtr& arg1,
                      const HostTensorPtr& output,
                      const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;

        switch (arg0->get_element_type())
        {
            TYPE_CASE(i8)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(i16)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(i32)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(i64)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(u8)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(u16)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(u32)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(u64)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(bf16)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(f32)(arg0, arg1, output, broadcast_spec);
            break;
            TYPE_CASE(f64)(arg0, arg1, output, broadcast_spec);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v1::Mod::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    return evaluate_mod(inputs[0], inputs[1], outputs[0], get_auto_broadcast());
}
