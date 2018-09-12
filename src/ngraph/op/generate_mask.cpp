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

#include "ngraph/op/generate_mask.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::GenerateMask::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 1)
    {
        //GenerateMask(const std::shared_ptr<Node>& training, const std::shared_ptr<Node>& activate, const Shape& shape, const element::Type& element_type, double prob, const std::shared_ptr<RNGState>& state) :
        return make_shared<GenerateMask>(
            new_args.at(0), new_args.at(1), m_shape, m_element_type, m_probability, m_state);
    }
    return make_shared<GenerateMask>(
        new_args.at(0), m_shape, m_element_type, m_probability, m_state);
}

void ngraph::op::GenerateMask::validate_and_infer_types()
{
    NGRAPH_ASSERT(shape_size(get_training_node()->get_shape()) == 1)
        << "Training node should be a scalar flag indicating a mode";
    set_output_type(0, m_element_type, m_shape);
}
