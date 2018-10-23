//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include "ngraph/runtime/cpu/op/halide_op.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> runtime::cpu::op::HalideOp::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<HalideOp>(new_args, m_ops, m_output_type, m_output_shape);
}

runtime::cpu::op::HalideOp::HalideOp(const NodeVector& args,
                                     const std::list<std::shared_ptr<Node>>& ops,
                                     const element::Type& out_type,
                                     const Shape& out_shape)
    : Op("HalideOp", check_single_output_args(args))
    , m_ops(ops)
    , m_output_type(out_type)
    , m_output_shape(out_shape)
{
    constructor_validate_and_infer_types();
}

void runtime::cpu::op::HalideOp::validate_and_infer_types()
{
    set_output_type(0, m_output_type, m_output_shape);
}
