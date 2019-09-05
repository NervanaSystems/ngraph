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

#include "ngraph/op/product.hpp"

using namespace std;
using namespace ngraph;

const string op::v0::Product::type_name{"Product"};

op::v0::Product::Product(const Output<Node>& arg, const AxisSet& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

op::v0::Product::Product(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Product::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Product>(new_args.at(0), get_reduction_axes());
}

const string op::v1::ReduceProd::type_name{"Product"};

op::v1::ReduceProd::ReduceProd(const Output<Node>& arg,
                               const AxisSet& reduction_axes,
                               bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{keep_dims}
{
    constructor_validate_and_infer_types();
}

op::v1::ReduceProd::ReduceProd(const Output<Node>& arg,
                               const Output<Node>& reduction_axes,
                               bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{keep_dims}
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceProd::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReduceProd>(new_args.at(0), get_reduction_axes(), m_keep_dims);
}
