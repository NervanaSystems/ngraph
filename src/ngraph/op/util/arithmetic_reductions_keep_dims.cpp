//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::util::ArithmeticReductionKeepDims::ArithmeticReductionKeepDims(
    const ngraph::Output<ngraph::Node>& arg,
    const ngraph::Output<ngraph::Node>& reduction_axes,
    bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{keep_dims}
{
}

void op::util::ArithmeticReductionKeepDims::validate_and_infer_types()
{
    if (m_keep_dims)
    {
        auto reduction_axes = get_reduction_axes();
        auto input_shape = get_input_partial_shape(0);
        auto input_rank = input_shape.rank();
        PartialShape result_shape{PartialShape::dynamic()};

        if (input_rank.is_static())
            result_shape = PartialShape::dynamic(input_rank);

        if (input_rank.is_static() && reduction_axes_constant())
        {
            std::vector<Dimension> dims;
            for (auto axis : reduction_axes)
            {
                NODE_VALIDATION_CHECK(this,
                                      axis < size_t(input_rank),
                                      "Reduction axis (",
                                      axis,
                                      ") is out of bounds ",
                                      "(argument shape: ",
                                      input_shape,
                                      ", reduction axes: ",
                                      reduction_axes,
                                      ")");
            }
            for (size_t i = 0; i < size_t(input_rank); i++)
            {
                if (reduction_axes.count(i) == 0)
                {
                    dims.push_back(input_shape[i]);
                }
                else
                {
                    dims.push_back(Dimension{1});
                }
            }
            result_shape = PartialShape(dims);
        }
        set_input_is_relevant_to_shape(1);
        set_output_type(0, get_input_element_type(0), result_shape);
    }
    else
    {
        ArithmeticReduction::validate_and_infer_types();
    }
}
