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

#include "ngraph/frontend/fluid/operators/pool.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/max_pool.hpp"

using namespace std;
using namespace ngraph::fluid;

static size_t calculate_adaptive(size_t input_dim, size_t window_dim)
{
    return floor(input_dim / window_dim);
}

constexpr NodeTypeInfo Pool::type_info;

Pool::Pool(const Output<Node>& x,
           const Shape& window_shape,
           const Strides& window_movement_strides,
           const Shape& padding,
           const bool global_pooling,
           const bool ceil_mode,
           const bool exclusive,
           const bool adaptive,
           const string pooling_type)
    : FusedOp({x})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding(padding)
    , m_global_pooling(global_pooling)
    , m_ceil_mode(ceil_mode)
    , m_exclusive(exclusive)
    , m_adaptive(adaptive)
    , m_pooling_type(pooling_type)
{
    constructor_validate_and_infer_types();
}

NodeVector Pool::decompose_op() const
{
    auto x = input_value(0);
    auto x_shape = get_input_shape(0);
    Shape window_shape = get_window_shape();
    Strides strides = get_window_movement_strides();
    Shape padding = get_padding();
    bool global_pooling = get_global_pooling();
    bool exclusive = get_exclusive();
    bool adaptive = get_adaptive();
    string pooling_type = get_pooling_type();

    NODE_VALIDATION_CHECK(
        this, x_shape.size() - 2 == window_shape.size(), "Supporting 2d pooling only");

    if (global_pooling)
    {
        for (size_t i = 0; i < window_shape.size(); ++i)
        {
            padding[i] = 0;
            window_shape[i] = x_shape[i + 2];
        }
    }

    shared_ptr<Node> pool;

    if (pooling_type == "max")
    {
        pool = make_shared<op::MaxPool>(x, window_shape, strides, padding, padding);
    }
    else if (pooling_type == "avg")
    {
        if (adaptive)
        {
            if (x_shape.size() == 4)
            {
                strides[0] = calculate_adaptive(x_shape[2], window_shape[0]);
                strides[1] = calculate_adaptive(x_shape[3], window_shape[1]);
            }

            pool = make_shared<op::AvgPool>(x, window_shape, strides);
        }
        else
        {
            if (padding[0] == 0 && padding[1] == 0)
            {
                exclusive = false;
            }

            pool = make_shared<op::AvgPool>(x, window_shape, strides, padding, padding, !exclusive);
        }
    }
    else
    {
        throw ngraph_error("Unsupported pooling type");
    }

    return {pool};
}

shared_ptr<Node> Pool::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);

    return make_shared<Pool>(new_args.at(0),
                             get_window_shape(),
                             get_window_movement_strides(),
                             get_padding(),
                             get_global_pooling(),
                             get_ceil_mode(),
                             get_exclusive(),
                             get_adaptive(),
                             get_pooling_type());
}

void Pool::pre_validate_and_infer_types()
{
    auto shape = get_input_partial_shape(0);

    if (shape.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

constexpr NodeTypeInfo PoolGrad::type_info;

PoolGrad::PoolGrad(const Output<Node>& x,
                   const Output<Node>& output,
                   const Output<Node>& output_delta,
                   const Shape& window_shape,
                   const Strides& window_movement_strides,
                   const Shape& padding,
                   const bool global_pooling,
                   const bool exclusive,
                   const bool adaptive,
                   const string pooling_type)
    : FusedOp({x, output, output_delta})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding(padding)
    , m_global_pooling(global_pooling)
    , m_exclusive(exclusive)
    , m_adaptive(adaptive)
    , m_pooling_type(pooling_type)
{
    constructor_validate_and_infer_types();
}

void PoolGrad::pre_validate_and_infer_types()
{
    if (get_input_partial_shape(0).is_dynamic() || get_input_partial_shape(1).is_dynamic() ||
        get_input_partial_shape(2).is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> PoolGrad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<PoolGrad>(new_args.at(0),
                                 new_args.at(1),
                                 new_args.at(2),
                                 get_window_shape(),
                                 get_window_movement_strides(),
                                 get_padding(),
                                 get_global_pooling(),
                                 get_exclusive(),
                                 get_adaptive(),
                                 get_pooling_type());
}

NodeVector PoolGrad::decompose_op() const
{
    auto x = input_value(0);
    auto x_shape = get_input_shape(0);
    auto output = input_value(1);
    auto output_delta = input_value(2);
    Shape window_shape = get_window_shape();
    Strides strides = get_window_movement_strides();
    Shape padding = get_padding();
    bool global_pooling = get_global_pooling();
    bool exclusive = get_exclusive();
    bool adaptive = get_adaptive();
    string pooling_type = get_pooling_type();

    NODE_VALIDATION_CHECK(
        this, x_shape.size() - 2 == window_shape.size(), "Supporting 2d pooling only");

    if (global_pooling)
    {
        for (size_t i = 0; i < window_shape.size(); ++i)
        {
            padding[i] = 0;
            window_shape[i] = x_shape[i + 2];
        }
    }

    shared_ptr<Node> pool_grad;

    if (pooling_type == "max")
    {
        pool_grad = make_shared<op::MaxPoolBackprop>(
            x, output_delta, output, window_shape, strides, padding, padding);
    }
    else if (pooling_type == "avg")
    {
        if (adaptive && x_shape.size() == 4)
        {
            strides[0] = calculate_adaptive(x_shape[2], window_shape[0]);
            strides[1] = calculate_adaptive(x_shape[3], window_shape[1]);
        }

        else if (padding[0] == 0 && padding[1] == 0)
        {
            exclusive = false;
        }

        pool_grad = make_shared<op::AvgPoolBackprop>(
            x.get_shape(), output_delta, window_shape, strides, padding, padding, !exclusive);
    }
    else
    {
        throw ngraph_error("Unsupported pooling type");
    }

    return {pool_grad};
}
