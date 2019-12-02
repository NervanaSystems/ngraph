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
#include <cstring>
#include <numeric>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/frontend/fluid/operators/reduce_sum.hpp"
#include "ngraph/op/reduce_sum.hpp"

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
    NodeVector retval;
    // Use reduce_sum v1 to support keep_dim
    if (m_reduce_all)
    {
        auto input_shape = get_input_partial_shape(0);
        if (input_shape.is_dynamic())
        {
            throw ngraph_error("Input needs to have static shape to decompose");
        }
        auto shape = input_shape.to_shape();
        vector<int64_t> axes;
        iota(axes.begin(), axes.end(), 0);
        auto axes_node = make_shared<ngraph::op::Constant>(element::i64, shape, axes);
        auto node = make_shared<ngraph::op::v1::ReduceSum>(input_value(0), axes_node, m_keep_dim);
        retval.emplace_back(node);
    }
    else
    {
        auto axes_node =
            make_shared<ngraph::op::Constant>(element::i64, Shape(m_dim.size()), m_dim);
        auto node = make_shared<ngraph::op::v1::ReduceSum>(input_value(0), axes_node, m_keep_dim);
        retval.emplace_back(node);
    }
    return retval;
}

shared_ptr<Node> ReduceSum::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ReduceSum>(new_args.at(0), m_dim, m_reduce_all, m_keep_dim);
}

constexpr NodeTypeInfo ReduceSumGrad::type_info;

ReduceSumGrad::ReduceSumGrad(const Output<Node>& x,
                             const vector<int>& dim,
                             bool reduce_all,
                             bool keep_dim)
    : FusedOp({x})
    , m_dim(dim)
    , m_reduce_all(reduce_all)
    , m_keep_dim(keep_dim)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> ReduceSumGrad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReduceSumGrad>(new_args.at(0), m_dim, m_reduce_all, m_keep_dim);
}

NodeVector ReduceSumGrad::decompose_op() const
{
    return {};
}
