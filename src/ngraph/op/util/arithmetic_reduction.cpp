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

#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::util::ArithmeticReduction::ArithmeticReduction()
{
}

op::util::ArithmeticReduction::ArithmeticReduction(const std::shared_ptr<Node>& arg,
                                                   const AxisSet& reduction_axes)
    : Op(check_single_output_args({arg,
                                   op::Constant::create(element::i64,
                                                        Shape{reduction_axes.size()},
                                                        reduction_axes.to_vector())}))
{
    set_reduction_axes(reduction_axes);
}

op::util::ArithmeticReduction::ArithmeticReduction(const std::string& node_type,
                                                   const std::shared_ptr<Node>& arg,
                                                   const AxisSet& reduction_axes)
    : Op(node_type,
         check_single_output_args({arg,
                                   op::Constant::create(element::i64,
                                                        Shape{reduction_axes.size()},
                                                        reduction_axes.to_vector())}))
{
}

op::util::ArithmeticReduction::ArithmeticReduction(const std::string& node_type,
                                                   const std::shared_ptr<Node>& arg,
                                                   const std::shared_ptr<Node>& reduction_axes)
    : Op(node_type, check_single_output_args({arg, reduction_axes}))
{
}

void op::util::ArithmeticReduction::set_reduction_axes(const AxisSet& reduction_axes)
{
    m_reduction_axes = reduction_axes;
}

void op::util::ArithmeticReduction::validate_and_infer_types()
{
    if (auto const_op = dynamic_pointer_cast<op::Constant>(get_argument(1)))
    {
        m_reduction_axes = const_op->get_axis_set_val();
    }

    auto input_shape = get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    PartialShape result_shape{PartialShape::dynamic()};

    if (input_rank.is_static())
    {
        std::vector<Dimension> dims;

        for (auto axis : m_reduction_axes)
        {
            NODE_VALIDATION_CHECK(this,
                                  axis < size_t(input_rank),
                                  "Reduction axis (",
                                  axis,
                                  ") is out of bounds ",
                                  "(argument shape: ",
                                  input_shape,
                                  ", reduction axes: ",
                                  m_reduction_axes,
                                  ")");
        }

        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            if (m_reduction_axes.count(i) == 0)
            {
                dims.push_back(input_shape[i]);
            }
        }

        result_shape = PartialShape(dims);
    }

    set_output_type(0, get_input_element_type(0), result_shape);
}
