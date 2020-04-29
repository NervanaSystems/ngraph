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
#include "ngraph/op/fused/clamp.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/runtime/reference/clamp.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Clamp::type_info;

namespace
{
    template <element::Type_t ET>
    bool evaluate(
        const HostTensorPtr& arg, const HostTensorPtr& out, double min, double max, size_t count)
    {
        runtime::reference::clamp(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), min, max, count);
        return true;
    }

    bool evaluate_clamp(
        const HostTensorPtr& arg, const HostTensorPtr& out, double min, double max, size_t count)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(f32)(arg, out, min, max, count);
            break;
            TYPE_CASE(f64)(arg, out, min, max, count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Clamp::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_clamp(
        inputs[0], outputs[0], get_min(), get_max(), shape_size(get_output_shape(0)));
}

op::Clamp::Clamp(const Output<Node>& data, const double min, const double max)
    : FusedOp({data})
    , m_min{min}
    , m_max{max}
{
    constructor_validate_and_infer_types();
}

void op::Clamp::pre_validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0) == element::f64 ||
                              get_input_element_type(0) == element::f32,
                          "Clamp input must be a floating point input, either f64 or f32");

    NODE_VALIDATION_CHECK(
        this, m_min < m_max, "The 'min' parameter needs to be less than 'max' for Clamp");
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

NodeVector op::Clamp::decompose_op() const
{
    const auto data = input_value(0);
    const auto data_shape = data.get_shape();

    const auto clamp_min = builder::make_constant(data.get_element_type(), data_shape, m_min);
    const auto clamp_max = builder::make_constant(data.get_element_type(), data_shape, m_max);

    return {std::make_shared<ngraph::op::Minimum>(
        clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
}

shared_ptr<Node> op::Clamp::clone_with_new_inputs(const OutputVector& new_args) const
{
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the Clamp op but got ",
                          new_args.size());

    return make_shared<Clamp>(new_args.at(0), m_min, m_max);
}

bool op::Clamp::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
    return true;
}
