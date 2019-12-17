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

#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::DynReshape::type_info;

op::v0::DynReshape::DynReshape(const Output<Node>& arg, const Output<Node>& pattern, bool zero_flag)
    : Op({arg, pattern})
    , m_zero_flag(zero_flag)
{
    constructor_validate_and_infer_types();
}

void op::v0::DynReshape::validate_and_infer_types()
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

    if (auto const_shape = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        std::vector<int64_t> out_shape_val = const_shape->get_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              std::none_of(out_shape_val.begin(),
                                           out_shape_val.end(),
                                           [](int64_t v) { return v < -1; }),
                              "Dim size cannot be less than -1 ");

        int zero_dims = std::count_if(
            out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == 0; });
        int negative_dims = std::count_if(
            out_shape_val.begin(), out_shape_val.end(), [](int64_t v) { return v == -1; });
        NODE_VALIDATION_CHECK(this,
                              negative_dims <= 1,
                              "More than one dimension has size of -1 (",
                              negative_dims,
                              ")");

        if (!(zero_dims && m_zero_flag) && !negative_dims)
        {
            set_output_type(0, get_input_element_type(0), const_shape->get_shape_val());
        }
        else
        {
            std::vector<Dimension> partial_shape(static_cast<size_t>(output_rank));
            // Replace zeros and negatives with Dynamic dimensions as needed
            std::transform(out_shape_val.begin(),
                           out_shape_val.end(),
                           partial_shape.begin(),
                           [&](const int64_t& v) {
                               return (v < 0)
                                          ? Dimension()
                                          : ((v == 0 && m_zero_flag) ? Dimension() : Dimension(v));
                           });

            if (get_input_partial_shape(0).is_static())
            {
                size_t output_elements = 1;
                int negative_dim = -1;

                auto input_shape = get_input_partial_shape(0).to_shape();
                size_t input_elements = shape_size(input_shape);
                for (size_t i = 0; i < static_cast<size_t>(output_rank); i++)
                {
                    if (out_shape_val[i] == 0 && m_zero_flag)
                    {
                        // Copy input_shape[i] for zero values
                        NODE_VALIDATION_CHECK(
                            this, i < input_shape.size(), "'0' dimension is out of range");
                        partial_shape[i] = Dimension(input_shape[i]);
                        output_elements *= input_shape[i];
                    }
                    else if (out_shape_val[i] == -1)
                    {
                        negative_dim = i;
                    }
                    else
                    {
                        output_elements *= out_shape_val[i];
                    }
                }

                if (negative_dim != -1)
                {
                    // Infer size such that number of output elements matches
                    // input elements
                    if (output_elements == 0)
                    {
                        // TODO(amprocte): Decide if this is desired behavior here. (NumPy seems
                        // to fail.)
                        NODE_VALIDATION_CHECK(this,
                                              input_elements == 0,
                                              "Cannot infer '-1' dimension with zero-size output "
                                              "dimension unless at least one input dimension is "
                                              "also zero-size");
                        partial_shape[negative_dim] = Dimension(0);
                    }
                    else
                    {
                        NODE_VALIDATION_CHECK(
                            this,
                            input_elements % output_elements == 0,
                            "Non-'-1' output dimensions do not evenly divide the input dimensions");
                        partial_shape[negative_dim] = Dimension(input_elements / output_elements);
                    }
                }
            }
            set_output_type(0, get_input_element_type(0), PartialShape(partial_shape));
        }
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(output_rank));
    }
}

shared_ptr<Node> op::v0::DynReshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::DynReshape>(new_args.at(0), new_args.at(1), m_zero_flag);
}

void op::v0::DynReshape::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                           const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for DynReshape");
}
