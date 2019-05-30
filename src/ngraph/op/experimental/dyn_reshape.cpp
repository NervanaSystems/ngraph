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

#include <algorithm>
#include <iostream>

#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"

using namespace std;
using namespace ngraph;

op::DynReshape::DynReshape(const shared_ptr<Node>& arg, const shared_ptr<Node>& pattern)
    : Op("DynReshape", check_single_output_args({arg, pattern}))
{
    constructor_validate_and_infer_types();
}

void op::DynReshape::validate_and_infer_types()
{
    auto pattern_et = get_input_element_type(1);
    // check data types
    NODE_VALIDATION_CHECK(
        this, pattern_et.compatible(element::Type_t::i64), "Pattern must have element type i64.");

    // check shapes
    const PartialShape& pattern_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          pattern_shape.rank().compatible(1),
                          "Pattern shape must have rank 1, got ",
                          pattern_shape.rank(),
                          ".");
    Rank output_rank = pattern_shape.rank().is_dynamic() ? Rank::dynamic() : pattern_shape[0];

    set_input_is_relevant_to_shape(1);

    if (auto const_shape = dynamic_pointer_cast<op::Constant>(get_argument(1)))
    {
        std::vector<int64_t> out_shape_val = const_shape->get_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              std::none_of(out_shape_val.begin(),
                                           out_shape_val.end(),
                                           [](int64_t v) { return v < -1; }),
                              "Dim size cannot be less than -1 ",
                              out_shape_val);

        int zero_dims = std::count_if(
            out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == 0; });
        int negative_dims = std::count_if(
            out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == -1; });
        NODE_VALIDATION_CHECK(this,
                              negative_dims <= 1,
                              "More than one dimension has size of -1 (",
                              negative_dims,
                              ")");

        if (!zero_dims && !negative_dims)
        {
            set_output_type(0, get_input_element_type(0), const_shape->get_shape_val());
        }
        else if (!get_input_partial_shape(0).is_static())
        {
            // We need input shape to determine output shape in the presence of
            // zero and negative sized dims.
            set_output_type(0, get_input_element_type(0), PartialShape::dynamic(output_rank));
        }
        else
        {
            auto input_shape = get_input_partial_shape(0).to_shape();
            auto input_elements = shape_size(input_shape);

            Shape output_shape(input_shape.size());
            size_t output_elements = 1;
            int negative_dim = -1;
            for (size_t i = 0; i < input_shape.size(); i++)
            {
                if (out_shape_val[i] > 0)
                {
                    output_shape[i] = out_shape_val[i];
                    output_elements *= output_shape[i];
                }
                else if (out_shape_val[i] == 0)
                {
                    output_shape[i] = input_shape[i];
                    output_elements *= output_shape[i];
                }
                else
                {
                    NGRAPH_CHECK(negative_dim == -1);
                    negative_dim = i;
                }
            }
            if (negative_dim != -1)
            {
                // Infer size such that number of output elements matches
                // input elements
                NGRAPH_CHECK(input_elements % output_elements == 0);
                output_shape[negative_dim] = input_elements / output_elements;
            }
            set_output_type(0, get_input_element_type(0), output_shape);
        }
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(output_rank));
    }
}

shared_ptr<Node> op::DynReshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynReshape>(new_args.at(0), new_args.at(1));
}

void op::DynReshape::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for DynReshape");
}
