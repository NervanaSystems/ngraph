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

#include <cmath>
#include <cstring>
#include <numeric>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/frontend/fluid/operators/reduce_sum.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph::fluid;

constexpr NodeTypeInfo ReduceSum::type_info;

ReduceSum::ReduceSum(const Output<Node>& x, const vector<int>& dim, bool reduce_all, bool keep_dim)
    : FusedOp({x})
    , m_dim(dim)
    , m_reduce_all(reduce_all)
    , m_keep_dim(keep_dim)
{
    constructor_validate_and_infer_types();
}

NodeVector ReduceSum::decompose_op() const
{
    auto shape = get_input_partial_shape(0);
    if (shape.is_dynamic())
    {
        throw ngraph_error("Input needs to have static shape to decompose");
    }
    auto input_shape = shape.to_shape();
    int input_rank = static_cast<int>(input_shape.size());
    NodeVector retval;
    vector<size_t> axes;
    // Use reduce_sum v1 to support keep_dim
    if (m_reduce_all)
    {
        for (size_t axis = 0; axis < input_rank; axis++)
        {
            axes.emplace_back(axis);
        }
    }
    else
    {
        for (int axis : m_dim)
        {
            axes.emplace_back(axis < 0 ? static_cast<size_t>(axis + input_rank)
                                       : static_cast<size_t>(axis));
        }
    }
    auto axes_node = make_shared<ngraph::op::Constant>(element::i64, Shape{axes.size()}, axes);
    auto node = make_shared<ngraph::op::v1::ReduceSum>(input_value(0), axes_node, m_keep_dim);
    retval.emplace_back(node);
    return retval;
}

void ReduceSum::pre_validate_and_infer_types()
{
    auto shape = get_input_partial_shape(0);

    if (shape.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> ReduceSum::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReduceSum>(new_args.at(0), m_dim, m_reduce_all, m_keep_dim);
}

constexpr NodeTypeInfo ReduceSumGrad::type_info;

ReduceSumGrad::ReduceSumGrad(const Output<Node>& x,
                             const Output<Node>& y,
                             const vector<int>& dim,
                             bool reduce_all,
                             bool keep_dim)
    : FusedOp({x, y})
    , m_dim(dim)
    , m_reduce_all(reduce_all)
    , m_keep_dim(keep_dim)
{
    constructor_validate_and_infer_types();
}

NodeVector ReduceSumGrad::decompose_op() const
{
    auto x_shape = get_input_partial_shape(0);
    auto y_shape = get_input_partial_shape(1);
    if (x_shape.is_dynamic() || y_shape.is_dynamic())
    {
        throw ngraph_error("All input needs to have static shape to decompose");
    }
    auto input_shape = x_shape.to_shape();
    int input_rank = static_cast<int>(input_shape.size());
    NodeVector retval;
    vector<size_t> axes;
    if (m_reduce_all)
    {
        for (size_t axis = 0; axis < input_rank; axis++)
        {
            axes.emplace_back(axis);
        }
    }
    else
    {
        for (int axis : m_dim)
        {
            axes.emplace_back(axis < 0 ? static_cast<size_t>(axis + input_rank)
                                       : static_cast<size_t>(axis));
        }
    }
    AxisSet red_axes(axes);
    auto grad = input_value(1);
    // squeeze kept dim in y
    if (m_keep_dim)
    {
        auto grad_shape = y_shape.to_shape();
        AxisVector axis_vec(grad_shape.size());
        iota(axis_vec.begin(), axis_vec.end(), 0);
        for (size_t axis : axes)
        {
            grad_shape[axis] = 0;
        }
        vector<size_t> squeezed;
        for (size_t dim : grad_shape)
        {
            if (dim != 0)
            {
                squeezed.emplace_back(dim);
            }
        }
        Shape squeezed_grad_shape(squeezed);
        grad = make_shared<ngraph::op::v0::Reshape>(grad, axis_vec, squeezed_grad_shape);
    }
    // broadcast the reduced axes
    auto node = make_shared<ngraph::op::v0::Broadcast>(grad, input_shape, red_axes);
    retval.emplace_back(node);
    return retval;
}

void ReduceSumGrad::pre_validate_and_infer_types()
{
    auto shape = get_input_partial_shape(0);

    if (shape.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> ReduceSumGrad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReduceSumGrad>(
        new_args.at(0), new_args.at(1), m_dim, m_reduce_all, m_keep_dim);
}
