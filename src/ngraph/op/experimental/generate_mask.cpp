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

#include "ngraph/op/experimental/generate_mask.hpp"

using namespace std;
using namespace ngraph;

op::GenerateMask::GenerateMask(const std::shared_ptr<Node>& training,
                               const Shape& shape,
                               const element::Type& element_type,
                               unsigned int seed,
                               double prob)
    : Op("GenerateMask", check_single_output_args({training}))
    , m_shape(shape)
    , m_element_type(element_type)
    , m_seed(seed)
    , m_probability(prob)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::GenerateMask::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GenerateMask>(
        new_args.at(0), m_shape, m_element_type, m_seed, m_probability);
}

void ngraph::op::GenerateMask::validate_and_infer_types()
{
    NODE_VALIDATION_ASSERT(this, get_input_partial_shape(0).compatible(PartialShape{}))
        << "Training node should be a scalar flag indicating a mode";

    NODE_VALIDATION_ASSERT(this, m_element_type.is_static())
        << "Output element type must not be dynamic.";

    set_output_type(0, m_element_type, m_shape);
}
