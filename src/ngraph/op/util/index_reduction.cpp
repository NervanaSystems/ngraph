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
    , m_index_element_type(index_element_type)
{
    constructor_validate_and_infer_types();
}

void op::util::IndexReduction::validate_and_infer_types()
{
    const PartialShape& arg_shape = get_input_partial_shape(0);
    Rank rank = arg_shape.rank();

    NODE_VALIDATION_ASSERT(this, rank.is_dynamic() || size_t(rank) >= 1)
        << "Argument rank is zero.";
    NODE_VALIDATION_ASSERT(this, rank.is_dynamic() || m_axis < size_t(rank))
        << "Reduction axis (" << m_axis << ") is not less than argument rank (" << rank << ").";
    NODE_VALIDATION_ASSERT(
        this, m_index_element_type == element::i32 || m_index_element_type == element::i64)
        << "Index element is neither i64 or i32.";

    PartialShape output_shape{PartialShape::dynamic()};

    if (!rank.is_dynamic())
    {
        std::vector<Dimension> output_dims(size_t(rank) - 1);
        size_t j = 0;

        for (size_t i = 0; i < size_t(rank) - 1; i++)
        {
            if (j == m_axis)
            {
                j++;
            }
            output_dims[i] = arg_shape[j++];
        }

        output_shape = PartialShape(output_dims);
    }

    set_output_type(0, m_index_element_type, output_shape);
}

void op::util::IndexReduction::generate_adjoints(autodiff::Adjoints& adjoints,
                                                 const NodeVector& deltas)
{
    throw ngraph_error("Forward-propagation-only operation");
}
