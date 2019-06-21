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

#include "ngraph/op/sequence_repeat.hpp"

using namespace std;
using namespace ngraph;

const string op::SequenceRepeat::type_name("SequenceRepeat");

op::SequenceRepeat::SequenceRepeat(const Output<Node>& value)
    : Op({value})
{
}

void op::SequenceRepeat::validate_and_infer_types()
{
    PartialShape result_shape;
    const PartialShape& value_shape = get_input_partial_shape(0);
    if (value_shape.is_dynamic())
    {
        result_shape = PartialShape::dynamic();
    }
    else
    {
        vector<Dimension> result_dimensions;
        result_dimensions.push_back(Dimension::dynamic());
        for (size_t i = 0; i < static_cast<size_t>(value_shape.rank()); ++i)
        {
            result_dimensions.push_back(value_shape[i]);
        }
        result_shape = PartialShape(result_dimensions);
    }
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), result_shape);
}

shared_ptr<Node> op::SequenceRepeat::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SequenceRepeat>(new_args.at(0));
}
