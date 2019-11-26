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

#include "ngraph/op/group_convolution.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::GroupConvolution::type_info;
shared_ptr<Node> op::v1::GroupConvolution::get_default_value() const
{
    return op::Constant::create(get_element_type(), get_shape(), {0});
}

op::v1::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
                                           const Output<Node>& filters,
                                           const Strides& strides,
                                           const CoordinateDiff& pads_begin,
                                           const CoordinateDiff& pads_end,
                                           const Strides& dilations,
                                           const PadType& auto_pad)
    : Op({data_batch, filters})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
{
    constructor_validate_and_infer_types();
}

void op::v1::GroupConvolution::validate_and_infer_types()
{
    const PartialShape& data_batch_shape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_shape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    if (m_strides.size() == 0)
    {
        m_strides = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_dilations.size() == 0)
    {
        m_dilations = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (m_pads_begin.size() == 0)
    {
        m_pads_begin = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_pads_end.size() == 0)
    {
        m_pads_end = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        if (data_batch_shape.is_static() && filters_shape.is_static())
        {
            m_pads_begin.clear();
            m_pads_end.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 3); // Remove {G,O,I}
            infer_auto_padding(data_batch_shape.to_shape(),
                               filter_shape,
                               m_strides,
                               m_dilations,
                               m_auto_pad,
                               m_pads_end,
                               m_pads_begin);
        }
    }

    element::Type result_et;
    PartialShape result_shape;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

    result_shape = infer_convolution_forward(this,
                                             data_batch_shape,
                                             Strides(m_strides.size(), 1), // dummy data dilations
                                             m_pads_begin,
                                             m_pads_end,
                                             filters_shape,
                                             m_strides,
                                             m_dilations);

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::v1::GroupConvolution::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::GroupConvolution>(new_args.at(0),
                                             new_args.at(1),
                                             m_strides,
                                             m_pads_begin,
                                             m_pads_end,
                                             m_dilations,
                                             m_auto_pad);
}

void op::v1::GroupConvolution::generate_adjoints(autodiff::Adjoints& adjoints,
                                                 const NodeVector& deltas)
{
    ngraph_error("Not Yet Implemented");
}

constexpr NodeTypeInfo op::v1::GroupConvolutionBackpropData::type_info;

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(
    const Output<Node>& data,
    const Output<Node>& filters,
    const Output<Node>& output_shape,
    const Strides& strides,
    const CoordinateDiff& pads_begin,
    const CoordinateDiff& pads_end,
    const Strides& dilations,
    const PadType& auto_pad,
    const CoordinateDiff& output_padding)
    : Op({data, filters, output_shape})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
    , m_output_padding(output_padding)
{
    constructor_validate_and_infer_types();
}

op::v1::GroupConvolutionBackpropData::GroupConvolutionBackpropData(
    const Output<Node>& data,
    const Output<Node>& filters,
    const Strides& strides,
    const CoordinateDiff& pads_begin,
    const CoordinateDiff& pads_end,
    const Strides& dilations,
    const PadType& auto_pad,
    const CoordinateDiff& output_padding)
    : Op({data, filters})
    , m_strides(strides)
    , m_dilations(dilations)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_auto_pad(auto_pad)
    , m_output_padding(output_padding)
{
    constructor_validate_and_infer_types();
}

const PartialShape op::v1::GroupConvolutionBackpropData::get_output_shape() const
{
    PartialShape shape{PartialShape::dynamic()};
    bool is_output_shape_present = get_inputs().size() == 3;
    if (is_output_shape_present)
    {
        if (auto const_op = as_type<op::Constant>(input_value(2).get_node()))
        {
            shape = const_op->get_shape_val();
        }
    }
    return shape;
}

void op::v1::GroupConvolutionBackpropData::set_output_shape(const Shape& shape)
{
    this->input(2).replace_source_output(
        op::Constant::create(element::i64, Shape{shape.size()}, shape)->output(0));
}

void op::v1::GroupConvolutionBackpropData::validate_and_infer_types()
{
    auto data_pshape = get_input_partial_shape(0);
    element::Type delta_et = get_input_element_type(0);
    const PartialShape& filters_pshape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    bool is_output_shape_present = get_inputs().size() == 3;
    PartialShape output_pshape = get_output_shape();

    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, delta_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        delta_et,
        ", filters element type: ",
        filters_et,
        ").");

    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        NODE_VALIDATION_CHECK(this,
                              is_output_shape_present,
                              "Selected Pad type: ",
                              m_auto_pad,
                              "requires an output_shape input which is missing.");
        if (output_pshape.is_static() && filters_pshape.is_static())
        {
            m_pads_begin.clear();
            m_pads_end.clear();
            auto filter_shape = filters_pshape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(output_pshape.to_shape(),
                               filter_shape,
                               m_strides,
                               m_dilations,
                               m_auto_pad,
                               m_pads_end,
                               m_pads_begin);
        }
    }

    PartialShape result_shape;
    if (is_output_shape_present)
    {
        set_input_is_relevant_to_shape(2);
        if (output_pshape.is_static() && data_pshape.is_static())
        {
            auto data_shape = data_pshape.to_shape();
            auto output_shape = output_pshape.to_shape();
            output_shape.insert(output_shape.begin(), data_shape.begin(), data_shape.begin() + 1);
            output_pshape = output_shape;
        }
    }
    else
    {
        if (filters_pshape.is_static() && data_pshape.is_static())
        {
            auto filters_shape = filters_pshape.to_shape();
            auto data_shape = data_pshape.to_shape();

            Shape output_shape;
            auto data_spatial_rank = data_shape.size() - 3;
            auto output_padding = get_output_padding();
            if (output_padding.size() == 0)
            {
                output_padding.insert(output_padding.begin(), data_spatial_rank, 0);
            }
            for (size_t i = 0; i < data_spatial_rank; ++i)
            {
                size_t tmp = m_strides[i] * (data_shape[i + 2] - 1) +
                             ((filters_shape[i] + 2 - 1) * m_dilations[i] + 1) - m_pads_begin[i] -
                             m_pads_end[i] + output_padding[i];
                output_shape.push_back(tmp);
                output_pshape = output_shape;
            }
            output_shape.insert(output_shape.begin(), data_shape.begin(), data_shape.begin() + 1);
        }
    }

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, result_et, output_pshape);
}

void op::v1::GroupConvolutionBackpropData::generate_adjoints(autodiff::Adjoints& adjoints,
                                                             const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);
    const auto x_shape = x.get_shape();

    auto f = input_value(1);
    const auto f_shape = f.get_shape();

    auto data_conv = make_shared<op::v1::GroupConvolution>(
        delta, f, m_strides, m_pads_begin, m_pads_end, m_dilations, m_auto_pad);

    adjoints.add_delta(x, data_conv);

    Strides strides = m_dilations;
    CoordinateDiff pads_begin;
    CoordinateDiff pads_end;
    const Shape& filters_shape = get_input_shape(1);
    for (size_t i = 0; i < f_shape.size() - 2; i++)
    {
        ptrdiff_t pads_begin_backward =
            (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) - m_pads_begin[i];
        pads_begin.push_back(pads_begin_backward);

        ptrdiff_t pads_end_backward =
            (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) * m_dilations[i] +
            ((m_pads_begin[i] +
              (static_cast<size_t>(get_output_shape()[i + 2]) - 1) * m_strides[i] + m_pads_end[i] -
              (static_cast<ptrdiff_t>(filters_shape[i + 2]) - 1) * m_dilations[i]) %
             m_strides[i]) -
            m_pads_end[i];

        pads_end.push_back(pads_end_backward -
                           (pads_begin_backward + (x_shape[i + 2] - 1) * m_strides[i] +
                            pads_end_backward - (f_shape[i + 2] - 1) * m_dilations[i]) %
                               m_strides[i]);
    }

    auto swap_NC = [](const Output<Node>& n) {
        AxisVector ax_order = ngraph::get_default_order(n.get_shape());
        ax_order[0] = 1;
        ax_order[1] = 0;

        auto new_shape = n.get_shape();
        new_shape[0] = n.get_shape()[1];
        new_shape[1] = n.get_shape()[0];

        return make_shared<op::Reshape>(n, ax_order, new_shape);
    };

    delta = swap_NC(delta);
    x = swap_NC(x);

    shared_ptr<Node> filter_deconv_bprop = make_shared<op::v1::GroupConvolution>(
        x, delta, strides, pads_begin, pads_end, Strides(x.get_shape().size() - 3, 1), m_auto_pad);
    AxisSet axes;
    for (size_t i = 2; i < filter_deconv_bprop->get_shape().size(); ++i)
    {
        axes.insert(i);
    }
    filter_deconv_bprop = make_shared<ngraph::op::Reverse>(filter_deconv_bprop, axes);
    adjoints.add_delta(f, filter_deconv_bprop);
}

shared_ptr<Node>
    op::v1::GroupConvolutionBackpropData::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 3)
    {
        return make_shared<v1::GroupConvolutionBackpropData>(new_args.at(0),
                                                             new_args.at(1),
                                                             new_args.at(2),
                                                             m_strides,
                                                             m_pads_begin,
                                                             m_pads_end,
                                                             m_dilations,
                                                             m_auto_pad,
                                                             m_output_padding);
    }
    else
    {
        return make_shared<v1::GroupConvolutionBackpropData>(new_args.at(0),
                                                             new_args.at(1),
                                                             m_strides,
                                                             m_pads_begin,
                                                             m_pads_end,
                                                             m_dilations,
                                                             m_auto_pad,
                                                             m_output_padding);
    }
}
