// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include <sstream>

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace std;

std::string ngraph::Op::get_node_id() const
{
    stringstream ss;
    ss << get_op_class_name() << "_" << m_instance_id;
    return ss.str();
}

Node::ptr ngraph::op::abs(const Node::ptr& arg)
{
    return make_shared<AbsOp>(arg);
}

Node::ptr ngraph::op::add(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<AddOp>(arg0, arg1);
}

Node::ptr ngraph::op::ceiling(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<CeilingOp>(arg0, arg1);
}

// 'convert',
// 'convolution',

Node::ptr ngraph::op::divide(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<DivideOp>(arg0, arg1);
}

Node::ptr ngraph::op::exp(const Node::ptr& arg0)
{
    return make_shared<ExpOp>(arg0);
}

Node::ptr ngraph::op::floor(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<FloorOp>(arg0, arg1);
}

Node::ptr ngraph::op::log(const Node::ptr& arg0)
{
    return make_shared<LogOp>(arg0);
}

Node::ptr ngraph::op::maximum(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<MaximumOp>(arg0, arg1);
}

Node::ptr ngraph::op::minimum(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<MinimumOp>(arg0, arg1);
}

Node::ptr ngraph::op::multiply(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<MultiplyOp>(arg0, arg1);
}

Node::ptr ngraph::op::negative(const Node::ptr& arg0)
{
    return make_shared<NegativeOp>(arg0);
}

// 'pad',

Node::ptr ngraph::op::power(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<PowerOp>(arg0, arg1);
}

//'reduce',

Node::ptr ngraph::op::remainder(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<RemainderOp>(arg0, arg1);
}

Node::ptr ngraph::op::reshape(const Node::ptr& arg0, const Shape& shape)
{
    return make_shared<ReshapeOp>(arg0, shape);
}

//'reverse',
//'rng',
// 'select',
//'slice',

Node::ptr ngraph::op::subtract(const Node::ptr& arg0, const Node::ptr& arg1)
{
    return make_shared<SubtractOp>(arg0, arg1);
}

// 'transpose',
// 'while'
