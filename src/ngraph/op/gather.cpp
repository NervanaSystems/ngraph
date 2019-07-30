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

#include "ngraph/op/gather.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

static int PARAMS = 0;
static int INDICES = 1;

const string op::Gather::type_name{"Gather"};

shared_ptr<Node> op::Gather::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Gather>(new_args.at(PARAMS), new_args.at(INDICES), m_axis);
}

void op::Gather::validate_and_infer_types()
{
    element::Type result_et = get_input_element_type(PARAMS);
    element::Type indices_et = get_input_element_type(INDICES);

    const PartialShape& params_shape = get_input_partial_shape(PARAMS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    // params rank must be at least (axis + 1)
    // indices value must be in range [0, params.shape[axis]).
    // output rank is rank(params) + rank(indices) - 1
    NODE_VALIDATION_CHECK(this,
                          params_shape.rank().is_dynamic() ||
                              static_cast<size_t>(params_shape.rank()) >
                                  static_cast<size_t>(m_axis),
                          "params rank is expected to be at least axis + 1");

    PartialShape result_shape;
    if (params_shape.rank().is_static() && indices_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(static_cast<size_t>(params_shape.rank()) +
                                           static_cast<size_t>(indices_shape.rank()) - 1);
        size_t i = 0;
        for (; i < static_cast<size_t>(m_axis); i++)
        {
            result_dims[i] = params_shape[i];
        }
        for (size_t j = 0; j < static_cast<size_t>(indices_shape.rank()); i++, j++)
        {
            result_dims[i] = indices_shape[j];
        }
        for (size_t j = static_cast<size_t>(m_axis) + 1;
             j < static_cast<size_t>(params_shape.rank());
             i++, j++)
        {
            result_dims[i] = params_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}
