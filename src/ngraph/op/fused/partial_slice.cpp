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

#include <cmath>
#include <numeric>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/fused/partial_slice.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PartialSlice::type_info;
constexpr NodeTypeInfo op::PartialSliceBackprop::type_info;

op::PartialSlice::PartialSlice(const Output<Node>& data,
                               const AxisVector& axes,
                               const std::vector<int64_t>& lower_bounds,
                               const std::vector<int64_t>& upper_bounds,
                               const AxisVector& decrease_axes)
    : FusedOp({data})
    , m_axes(axes)
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_decrease_axes(decrease_axes)
{
    constructor_validate_and_infer_types();
}

// All input shape should be static by this point
NodeVector op::PartialSlice::decompose_op() const
{
    const PartialShape& data_shape = get_input_partial_shape(0);
    if (data_shape.is_dynamic())
    {
        throw ngraph_error("Data needs to have static shape to decompose");
    }
    auto input = input_value(0);
    auto input_shape = input.get_shape();
    auto axes = get_axes();
    auto starts = get_lower_bounds();
    auto ends = get_upper_bounds();
    ngraph::Coordinate ng_start, ng_end;
    int axis, start, end;
    for (size_t i = 0; i < input_shape.size(); ++i)
    {
        ng_start.push_back(0);
        ng_end.push_back(input_shape[i]);
    }
    for (size_t i = 0; i < axes.size(); ++i)
    {
        axis = input_shape[axes[i]];
        start = starts[i] < 0 ? (starts[i] + axis) : starts[i];
        end = ends[i] < 0 ? (ends[i] + axis) : ends[i];
        start = std::max(start, 0);
        end = std::max(end, 0);
        start = std::min(start, axis);
        end = std::min(end, axis);
        start = std::min(start, end);
        ng_start[axes[i]] = start;
        ng_end[axes[i]] = end;
    }
    auto out = std::make_shared<op::Slice>(input, ng_start, ng_end);

    auto out_shape = out->get_shape();

    std::vector<size_t> out_axis_vec(out_shape.size());
    std::iota(out_axis_vec.begin(), out_axis_vec.end(), 0);

    // paddle::platform::TrimTrailingSingularDims(&out_shape);
    auto out_dim = std::make_shared<op::Reshape>(
        out, ngraph::AxisVector(out_axis_vec), ngraph::Shape(out_shape));
    // Return output nodes
    return {out_dim};
}

shared_ptr<Node> op::PartialSlice::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<PartialSlice>(
        new_args.at(0), m_axes, m_lower_bounds, m_upper_bounds, m_decrease_axes);
}

void op::PartialSlice::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
}

void op::PartialSlice::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
}

op::PartialSliceBackprop::PartialSliceBackprop(const Output<Node>& data,
                                               const Output<Node>& dout,
                                               const AxisVector& axes,
                                               const std::vector<int64_t>& lower_bounds,
                                               const std::vector<int64_t>& upper_bounds)
    : FusedOp({data})
    , m_axes(axes)
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
{
    constructor_validate_and_infer_types();
}

// All input shape should be static by this point
NodeVector op::PartialSliceBackprop::decompose_op() const
{
    const PartialShape& data_shape = get_input_partial_shape(0);
    if (data_shape.is_dynamic())
    {
        throw ngraph_error("Data needs to have static shape to decompose");
    }
    NodeVector retval;
    return retval;
}

shared_ptr<Node> op::PartialSliceBackprop::copy_with_new_args(const NodeVector& new_args) const
{
}

void op::PartialSliceBackprop::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
}
