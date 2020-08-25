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
#include "ngraph/op/clamp.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/runtime/reference/clamp.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Clamp::type_info;

namespace
{
    template <element::Type_t ET, typename T>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, T min, T max, size_t count)
    {
        runtime::reference::clamp<T>(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), min, max, count);
        return true;
    }

    bool evaluate_clamp(const HostTensorPtr& arg, const HostTensorPtr& out, const op::v0::Clamp* op)
    {
        bool rc = true;
        size_t count = shape_size(op->get_input_shape(0));
        switch (arg->get_element_type())
        {
            TYPE_CASE(i8)(arg, out, op->get_min<int8_t>(), op->get_max<int8_t>(), count);
            break;
            TYPE_CASE(i16)(arg, out, op->get_min<int16_t>(), op->get_max<int16_t>(), count);
            break;
            TYPE_CASE(i32)(arg, out, op->get_min<int32_t>(), op->get_max<int32_t>(), count);
            break;
            TYPE_CASE(i64)(arg, out, op->get_min<int64_t>(), op->get_max<int64_t>(), count);
            break;
            TYPE_CASE(u8)(arg, out, op->get_min<uint8_t>(), op->get_max<uint8_t>(), count);
            break;
            TYPE_CASE(u16)(arg, out, op->get_min<uint16_t>(), op->get_max<uint16_t>(), count);
            break;
            TYPE_CASE(u32)(arg, out, op->get_min<uint32_t>(), op->get_max<uint32_t>(), count);
            break;
            TYPE_CASE(u64)(arg, out, op->get_min<uint64_t>(), op->get_max<uint64_t>(), count);
            break;
            TYPE_CASE(f16)(arg, out, op->get_min<float16>(), op->get_max<float16>(), count);
            break;
            TYPE_CASE(bf16)(arg, out, op->get_min<bfloat16>(), op->get_max<bfloat16>(), count);
            break;
            TYPE_CASE(f32)(arg, out, op->get_min<float>(), op->get_max<float>(), count);
            break;
            TYPE_CASE(f64)(arg, out, op->get_min<double>(), op->get_max<double>(), count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Clamp::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    return evaluate_clamp(inputs[0], outputs[0], this);
}

op::v0::Clamp::Clamp(const Output<Node>& data, const double min, const double max)
    : FusedOp({data})
    , m_min{min}
    , m_max{max}
{
    constructor_validate_and_infer_types();
}

void op::v0::Clamp::pre_validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(
        this, m_min < m_max, "The 'min' parameter needs to be less than 'max' for Clamp");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

OutputVector op::v0::Clamp::decompose_op() const
{
    const auto data = input_value(0);
    const auto type = data.get_element_type();
    const auto shape = data.get_shape();

    shared_ptr<Node> clamp_min;
    shared_ptr<Node> clamp_max;

    switch (type)
    {
    case element::Type_t::i8:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<int8_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<int8_t>());
        break;
    }
    case element::Type_t::i16:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<int16_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<int16_t>());
        break;
    }
    case element::Type_t::i32:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<int32_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<int32_t>());
        break;
    }
    case element::Type_t::i64:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<int64_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<int64_t>());
        break;
    }
    case element::Type_t::u8:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<uint8_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<uint8_t>());
        break;
    }
    case element::Type_t::u16:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<uint16_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<uint16_t>());
        break;
    }
    case element::Type_t::u32:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<uint32_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<uint32_t>());
        break;
    }
    case element::Type_t::u64:
    {
        clamp_min = make_shared<op::v0::Constant>(type, shape, get_min<uint64_t>());
        clamp_max = make_shared<op::v0::Constant>(type, shape, get_max<uint64_t>());
        break;
    }
    case element::Type_t::f16:
    {
        clamp_min = builder::make_constant(type, shape, get_min<float16>());
        clamp_max = builder::make_constant(type, shape, get_max<float16>());
        break;
    }
    case element::Type_t::bf16:
    {
        clamp_min = builder::make_constant(type, shape, get_min<bfloat16>());
        clamp_max = builder::make_constant(type, shape, get_max<bfloat16>());
        break;
    }
    case element::Type_t::f32:
    {
        clamp_min = builder::make_constant(type, shape, get_min<float>());
        clamp_max = builder::make_constant(type, shape, get_max<float>());
        break;
    }
    case element::Type_t::f64:
    {
        clamp_min = builder::make_constant(type, shape, get_min<double>());
        clamp_max = builder::make_constant(type, shape, get_max<double>());
        break;
    }
    default: throw runtime_error("Unsupported data type in op Clamp"); break;
    }

    auto max = make_shared<op::v1::Maximum>(clamp_min, data);
    return {make_shared<op::v1::Minimum>(clamp_max, max)};
}

shared_ptr<Node> op::v0::Clamp::clone_with_new_inputs(const OutputVector& new_args) const
{
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the Clamp op but got ",
                          new_args.size());

    return make_shared<Clamp>(new_args.at(0), m_min, m_max);
}

bool op::v0::Clamp::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
    return true;
}
