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

#include <algorithm>
#include <iostream>

#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Reshape::type_info;

op::Reshape::Reshape(const Output<Node>& arg,
                     const AxisVector& input_order,
                     const Shape& output_shape)
    : Op({arg})
    , m_input_order(input_order)
    , m_output_shape(output_shape)
{
    constructor_validate_and_infer_types();
}

void op::Reshape::validate_and_infer_types()
{
    auto& input_shape = get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    // Check that the input axis order is a permutation of (0,...,n-1) for some n.
    for (size_t i = 0; i < m_input_order.size(); i++)
    {
        NODE_VALIDATION_CHECK(
            this,
            find(begin(m_input_order), end(m_input_order), i) != end(m_input_order),
            "Input axis order is not a permutation of argument's axis indices (axis order: ",
            m_input_order,
            ", argument shape: ",
            input_shape,
            ").");
    }

    // TODO(amprocte): should be possible to move around unknown dims in the input shape.
    if (input_rank.is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            m_input_order.size() == input_rank.get_length(),
            "Input axis order is not a permutation of argument's axis indices (axis order: ",
            m_input_order,
            ", argument shape: ",
            input_shape,
            ").");

        for (size_t i = 0; i < input_rank.get_length(); i++)
        {
            auto it = find(begin(m_input_order), end(m_input_order), i);
            NODE_VALIDATION_CHECK(
                this,
                it != end(m_input_order),
                "Input axis order is not a permutation of argument's axis indices (axis order: ",
                m_input_order,
                ", argument shape: ",
                input_shape,
                ").");
        }

        // TODO(amprocte): make a partial_shape_size() analogous to shape_size().
        Dimension input_shape_product = 1;
        for (size_t i = 0; i < input_rank.get_length(); i++)
        {
            input_shape_product *= input_shape[i];
        }

        if (input_shape_product.is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                input_shape_product.get_length() == shape_size(m_output_shape),
                "Product of output shape dimensions does not match product of argument shape "
                "dimensions ",
                "(output shape: ",
                m_output_shape,
                ", argument shape: ",
                input_shape,
                ").");
        }
    }

    if (!std::is_sorted(m_input_order.begin(), m_input_order.end()))
    {
        m_is_transpose = true;
    }
    set_output_type(0, get_input_element_type(0), m_output_shape);
}

shared_ptr<Node> op::Reshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Reshape>(new_args.at(0), m_input_order, m_output_shape);
}

void op::Reshape::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x_shape = get_input_shape(0);
    auto x_rank = x_shape.size();
    Shape permuted_x_shape(x_rank);
    AxisVector x_input_order(x_rank);
    bool is_permuted = false;
    for (size_t i = 0; i < x_rank; ++i)
    {
        size_t permuted_i = m_input_order[i];
        if (i != permuted_i)
        {
            is_permuted = true;
        }
        permuted_x_shape[i] = x_shape[permuted_i];
        x_input_order[permuted_i] = i;
    }
    AxisVector input_order(m_output_shape.size());
    for (size_t i = 0; i < m_output_shape.size(); i++)
    {
        input_order[i] = i;
    }
    auto reshape = make_shared<op::Reshape>(delta, input_order, permuted_x_shape);
    if (is_permuted)
    {
        reshape = make_shared<op::Reshape>(reshape, x_input_order, x_shape);
    }

    adjoints.add_delta(input_value(0), reshape);
}

constexpr NodeTypeInfo op::v1::Reshape::type_info;

op::v1::Reshape::Reshape(const Output<Node>& arg, const Output<Node>& pattern, bool zero_flag)
    : Op({arg, pattern})
    , m_special_zero(zero_flag)
{
    constructor_validate_and_infer_types();
}

bool op::v1::Reshape::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("special_zero", m_special_zero);
    return true;
}

void op::v1::Reshape::validate_and_infer_types()
{
    auto pattern_et = get_input_element_type(1);
    // check data types
    NODE_VALIDATION_CHECK(
        this, pattern_et.is_integral_number(), "Pattern must be an integral number.");

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
        std::vector<int64_t> out_shape_val = const_shape->cast_vector<int64_t>();
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

        if (!(zero_dims && m_special_zero) && !negative_dims)
        {
            set_output_type(0, get_input_element_type(0), const_shape->get_shape_val());
        }
        else
        {
            std::vector<Dimension> partial_shape(output_rank.get_length());
            // Replace zeros and negatives with Dynamic dimensions as needed
            std::transform(out_shape_val.begin(),
                           out_shape_val.end(),
                           partial_shape.begin(),
                           [&](const int64_t& v) {
                               return (v < 0) ? Dimension()
                                              : ((v == 0 && m_special_zero) ? Dimension()
                                                                            : Dimension(v));
                           });

            if (get_input_partial_shape(0).is_static())
            {
                size_t output_elements = 1;
                int negative_dim = -1;

                auto input_shape = get_input_partial_shape(0).to_shape();
                size_t input_elements = shape_size(input_shape);
                for (size_t i = 0; i < output_rank.get_length(); i++)
                {
                    if (out_shape_val[i] == 0 && m_special_zero)
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

shared_ptr<Node> op::v1::Reshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Reshape>(new_args.at(0), new_args.at(1), m_special_zero);
}

void op::v1::Reshape::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                        const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for Reshape");
}
