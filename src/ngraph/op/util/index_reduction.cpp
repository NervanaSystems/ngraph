//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <memory>

#include "ngraph/op/util/index_reduction.hpp"

using namespace std;
using namespace ngraph;

op::util::IndexReduction::IndexReduction(const std::string& node_type,
                                         const std::shared_ptr<Node>& arg,
                                         size_t axis,
                                         const element::Type& index_element_type)
    : Op(node_type, check_single_output_args({arg}))
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
    auto rank = arg->get_shape().size();

    NODE_VALIDATION_ASSERT(this, rank >= 1) << "Argument rank must be at least 1";
    NODE_VALIDATION_ASSERT(this, axis < rank) << "Axis " << axis << " is greater than rank of "
                                              << rank;
    NODE_VALIDATION_ASSERT(this,
                           index_element_type == element::i32 || index_element_type == element::i64)
        << "Index element type must be i64 or i32";

    Shape output_shape = arg->get_shape();
    output_shape.erase(output_shape.begin() + axis);

    set_output_type(0, index_element_type, output_shape);
}

void op::util::IndexReduction::generate_adjoints(autodiff::Adjoints& adjoints,
                                                 const NodeVector& deltas)
{
    throw ngraph_error("Forward-propagation-only operation");
}
