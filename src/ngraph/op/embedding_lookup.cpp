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

#include "ngraph/op/embedding_lookup.hpp"

using namespace std;
using namespace ngraph;

void op::EmbeddingLookup::validate_and_infer_types()
{
    element::Type result_et = get_input_element_type(1);

    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);

    NODE_VALIDATION_ASSERT(
        this, arg1_shape.rank().is_dynamic() || static_cast<size_t>(arg1_shape.rank()) == 2)
        << "weights are expected to be a matrix";

    PartialShape result_shape;
    if (arg0_shape.rank().is_static())
    {
        std::vector<Dimension> result_dims(static_cast<size_t>(arg0_shape.rank()) + 1);
        for (size_t i = 0; i < static_cast<size_t>(arg0_shape.rank()); i++)
        {
            result_dims[i] = arg0_shape[i];
        }

        result_dims[result_dims.size() - 1] =
            arg1_shape.rank().is_static() ? arg1_shape[1] : Dimension::dynamic();
        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::EmbeddingLookup::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<EmbeddingLookup>(new_args.at(0), new_args.at(1));
}
