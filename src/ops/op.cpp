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

std::shared_ptr<Node> ngraph::op::abs(const std::shared_ptr<Node>& arg)
{
    return make_shared<AbsOp>(arg);
}

std::shared_ptr<Node> ngraph::op::add(const std::shared_ptr<Node>& arg0,
                                      const std::shared_ptr<Node>& arg1)
{
    return make_shared<AddOp>(arg0, arg1);
}

std::shared_ptr<Node> ngraph::op::ceiling(const std::shared_ptr<Node>& arg0,
                                          const std::shared_ptr<Node>& arg1)
{
    return make_shared<CeilingOp>(arg0, arg1);
}

// 'convert',
// 'convolution',

std::shared_ptr<Node> ngraph::op::divide(const std::shared_ptr<Node>& arg0,
                                         const std::shared_ptr<Node>& arg1)
{
    return make_shared<DivideOp>(arg0, arg1);
}

std::shared_ptr<Node> ngraph::op::exp(const std::shared_ptr<Node>& arg0)
{
    return make_shared<ExpOp>(arg0);
}

std::shared_ptr<Node> ngraph::op::floor(const std::shared_ptr<Node>& arg0,
                                        const std::shared_ptr<Node>& arg1)
{
    return make_shared<FloorOp>(arg0, arg1);
}

std::shared_ptr<Node> ngraph::op::log(const std::shared_ptr<Node>& arg0)
{
    return make_shared<LogOp>(arg0);
}

std::shared_ptr<Node> ngraph::op::maximum(const std::shared_ptr<Node>& arg0,
                                          const std::shared_ptr<Node>& arg1)
{
    return make_shared<MaximumOp>(arg0, arg1);
}

std::shared_ptr<Node> ngraph::op::minimum(const std::shared_ptr<Node>& arg0,
                                          const std::shared_ptr<Node>& arg1)
{
    return make_shared<MinimumOp>(arg0, arg1);
}

std::shared_ptr<Node> ngraph::op::multiply(const std::shared_ptr<Node>& arg0,
                                           const std::shared_ptr<Node>& arg1)
{
    return make_shared<MultiplyOp>(arg0, arg1);
}

std::shared_ptr<Node> ngraph::op::negative(const std::shared_ptr<Node>& arg0)
{
    return make_shared<NegativeOp>(arg0);
}

// 'pad',

std::shared_ptr<Node> ngraph::op::power(const std::shared_ptr<Node>& arg0,
                                        const std::shared_ptr<Node>& arg1)
{
    return make_shared<PowerOp>(arg0, arg1);
}

//'reduce',

std::shared_ptr<Node> ngraph::op::remainder(const std::shared_ptr<Node>& arg0,
                                            const std::shared_ptr<Node>& arg1)
{
    return make_shared<RemainderOp>(arg0, arg1);
}

std::shared_ptr<Node> ngraph::op::reshape(const std::shared_ptr<Node>& arg0, const Shape& shape)
{
    return make_shared<ReshapeOp>(arg0, shape);
}

//'reverse',
//'rng',
// 'select',
//'slice',

std::shared_ptr<Node> ngraph::op::subtract(const std::shared_ptr<Node>& arg0,
                                           const std::shared_ptr<Node>& arg1)
{
    return make_shared<SubtractOp>(arg0, arg1);
}

// 'transpose',
// 'while'
