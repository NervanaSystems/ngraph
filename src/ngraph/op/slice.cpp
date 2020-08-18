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

#include "ngraph/op/slice.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/slice.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Slice::type_info;

op::v0::Slice::Slice(const Output<Node>& arg,
                     const Coordinate& lower_bounds,
                     const Coordinate& upper_bounds,
                     const Strides& strides)
    : Op({arg})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(strides)
{
    constructor_validate_and_infer_types();
}

op::v0::Slice::Slice(const Output<Node>& arg,
                     const Coordinate& lower_bounds,
                     const Coordinate& upper_bounds)
    : Op({arg})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(Strides())
{
    constructor_validate_and_infer_types();
}

void op::v0::Slice::validate_and_infer_types()
{
    // An empty stride vector with lower_bounds/upper_bounds filled in means that we need to
    // construct the default value.
    if (m_strides.size() == 0)
    {
        m_strides = Strides(m_lower_bounds.size(), 1);
    }

    NODE_VALIDATION_CHECK(this,
                          m_lower_bounds.size() == m_upper_bounds.size() &&
                              m_lower_bounds.size() == m_strides.size(),
                          "Ranks of lower bounds (",
                          m_lower_bounds,
                          "), upper bounds (",
                          m_upper_bounds,
                          ") and strides (",
                          m_strides,
                          ") do not match.");

    size_t output_rank = m_upper_bounds.size();

    for (size_t i = 0; i < output_rank; i++)
    {
        NODE_VALIDATION_CHECK(this,
                              m_lower_bounds[i] <= m_upper_bounds[i],
                              "Lower bound for slice is greater than upper bound at axis ",
                              i,
                              " (lower bounds: ",
                              m_lower_bounds,
                              ", upper bounds: ",
                              m_upper_bounds,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              m_strides[i] != 0,
                              "Stride for slice is zero at axis ",
                              i,
                              " (strides: ",
                              m_strides,
                              ").");
    }

    const PartialShape& input_shape = get_input_partial_shape(0);
    Dimension input_rank = input_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || input_rank.get_length() == output_rank,
                          "Input rank does not match the rank of the lower bounds (",
                          m_lower_bounds,
                          "), upper bounds (",
                          m_upper_bounds,
                          "), and strides (",
                          m_strides,
                          ").");

    std::vector<Dimension> result_dims(output_rank);

    for (size_t i = 0; i < output_rank; i++)
    {
        NODE_VALIDATION_CHECK(this,
                              input_rank.is_dynamic() || input_shape[i].is_dynamic() ||
                                  m_upper_bounds[i] <= input_shape[i].get_length(),
                              "Upper bound for slice at axis ",
                              i,
                              " is out of range ",
                              "(upper bounds: ",
                              m_upper_bounds,
                              ", argument shape: ",
                              input_shape,
                              ").");

        size_t result_axis_size = m_upper_bounds[i] - m_lower_bounds[i];
        result_axis_size =
            result_axis_size / m_strides[i] + ((result_axis_size % m_strides[i] == 0) ? 0 : 1);
        result_dims[i] = result_axis_size;
    }

    set_output_type(0, get_input_element_type(0), PartialShape{result_dims});
}

shared_ptr<Node> op::v0::Slice::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Slice>(new_args.at(0), m_lower_bounds, m_upper_bounds, m_strides);
}

void op::v0::Slice::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);

    adjoints.add_delta_to_slice(x, delta, m_lower_bounds, m_upper_bounds, m_strides);
}

namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0,
                         const HostTensorPtr& out,
                         const Coordinate& lower_bounds,
                         const Coordinate& upper_bounds,
                         const Strides& strides)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::slice<T>(arg0->get_data_ptr<ET>(),
                                     out->get_data_ptr<ET>(),
                                     arg0->get_shape(),
                                     lower_bounds,
                                     upper_bounds,
                                     strides,
                                     out->get_shape());
        return true;
    }

    bool evaluate_slice(const HostTensorPtr& arg0,
                        const HostTensorPtr& out,
                        const Coordinate& lower_bounds,
                        const Coordinate& upper_bounds,
                        const Strides& strides)
    {
        bool rc = true;

        switch (arg0->get_element_type())
        {
            TYPE_CASE(boolean)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(i8)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(i16)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(i32)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(i64)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(u8)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(u16)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(u32)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(u64)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(bf16)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(f16)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(f32)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
            TYPE_CASE(f64)(arg0, out, lower_bounds, upper_bounds, strides);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Slice::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    return evaluate_slice(inputs[0], outputs[0], m_lower_bounds, m_upper_bounds, m_strides);
}
