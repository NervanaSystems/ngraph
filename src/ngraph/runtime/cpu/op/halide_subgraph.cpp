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

#include "ngraph/runtime/cpu/op/halide_subgraph.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node>
    runtime::cpu::op::HalideSubgraph::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<HalideSubgraph>(new_args, ops, output_type, output_shape);
}

runtime::cpu::op::HalideSubgraph::HalideSubgraph(const NodeVector& args,
                                                 const std::list<std::shared_ptr<Node>>& ops,
                                                 const element::Type& out_type,
                                                 const Shape& out_shape)
    : Op("HalideSubgraph", check_single_output_args(args))
    , ops(ops)
    , output_type(out_type)
    , output_shape(out_shape)
{
    constructor_validate_and_infer_types();
}

void runtime::cpu::op::HalideSubgraph::validate_and_infer_types()
{
    set_output_type(0, output_type, output_shape);
}
