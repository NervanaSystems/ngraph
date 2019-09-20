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

#include "ngraph/runtime/cpu/op/halide_op.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> runtime::cpu::op::HalideOp::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<HalideOp>(as_output_vector(new_args), m_ops, m_output_type, m_output_shape);
}

constexpr NodeTypeInfo runtime::cpu::op::HalideOp::type_info;

runtime::cpu::op::HalideOp::HalideOp(const OutputVector& args,
                                     const std::list<Output<Node>>& ops,
                                     const element::Type& out_type,
                                     const Shape& out_shape)
    : Op(args)
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
