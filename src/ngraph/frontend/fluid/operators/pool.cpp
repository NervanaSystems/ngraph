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

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/frontend/fluid/operators/pool.hpp"

using namespace std;
using namespace ngraph::fluid;

constexpr NodeTypeInfo Pool::type_info;

Pool::Pool(const Output<Node>& x,
           const Shape& window_shape,
           const Strides& window_movement_strides,
           const Shape& padding,
           const bool global_pooling,
           const string pooling_type)
    : FusedOp({x})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding(padding)
    , m_global_pooling(global_pooling)
    , m_pooling_type(pooling_type)
{
    constructor_validate_and_infer_types();
}

NodeVector Pool::decompose_op() const
{
    auto x = input_value(0);
    auto x_shape = get_input_shape(0);

    NODE_VALIDATION_CHECK(
        this, x_shape.size() - 2 == m_window_shape.size(), "Supporting 2d pooling only");

    return {};
}

shared_ptr<Node> Pool::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<Pool>(new_args.at(0),
                             m_window_shape,
                             m_window_movement_strides,
                             m_padding,
                             m_global_pooling,
                             m_pooling_type);
}

void Pool::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    if (input_pshape.is_dynamic())
    {
        set_output_type(0, input_element_type, input_pshape);
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
                   const string pooling_type)
    : FusedOp({x, output, output_delta})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding(padding)
    , m_global_pooling(global_pooling)
    , m_pooling_type(pooling_type)
{
    constructor_validate_and_infer_types();
}

void PoolGrad::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
}

shared_ptr<Node> PoolGrad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<PoolGrad>(new_args.at(0),
                                 new_args.at(1),
                                 new_args.at(2),
                                 m_window_shape,
                                 m_window_movement_strides,
                                 m_padding,
                                 m_global_pooling,
                                 m_pooling_type);
}

NodeVector PoolGrad::decompose_op() const
{
    return {};
}
