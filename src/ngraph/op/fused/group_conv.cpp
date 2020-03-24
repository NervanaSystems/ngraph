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

#include <numeric>

#include "group_conv.hpp"

#include "ngraph/builder/split.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
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
    : FusedOp({data_batch, filters})
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
    const PartialShape& data_batch_pshape = get_input_partial_shape(0);
    element::Type data_batch_et = get_input_element_type(0);
    const PartialShape& filters_pshape = get_input_partial_shape(1);
    element::Type filters_et = get_input_element_type(1);

    PartialShape result_shape{PartialShape::dynamic()};

    // we need to adjust filters_shape to reuse helpers for normal convolution
    if (filters_pshape.is_static() && data_batch_pshape.is_static())
    {
        auto filters_shape = filters_pshape.to_shape();
        auto groups = filters_shape[0];
        filters_shape[1] *= groups;
        filters_shape.erase(filters_shape.begin());
        auto data_batch_shape = data_batch_pshape.to_shape();
        data_batch_shape[1] /= groups;

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
            m_pads_begin.clear();
            m_pads_end.clear();
            infer_auto_padding(
                data_batch_shape,
                Shape(filters_shape.begin() + 2, filters_shape.end()), // Remove {O,I}
                m_strides,
                m_dilations,
                m_auto_pad,
                m_pads_end,
                m_pads_begin);
        }

        result_shape =
            infer_convolution_forward(this,
                                      data_batch_shape,
                                      Strides(m_strides.size(), 1), // dummy data dilations
                                      m_pads_begin,
                                      m_pads_end,
                                      filters_shape,
                                      m_strides,
                                      m_dilations);
    }
    element::Type result_et;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

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
                                                 const OutputVector& deltas)
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
    : FusedOp({data, filters, output_shape})
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
    : FusedOp({data, filters})
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
    PartialShape shape{vector<Dimension>(m_strides.size() + 2)};
    auto data_pshape = get_input_partial_shape(0);
    if (data_pshape.rank().is_static())
    {
        shape[0] = data_pshape[0]; // N
    }
    auto filters_pshape = get_input_partial_shape(1);
    if (filters_pshape.rank().is_static())
    {
        shape[1] = filters_pshape[1]; // C
    }
    bool is_output_shape_present = get_inputs().size() == 3;
    if (is_output_shape_present)
    {
        if (auto const_op = as_type<op::Constant>(input_value(2).get_node()))
        {
            auto output_shape = const_op->get_shape_val();
            // Populate spatials
            for (int i = 0; i < output_shape.size(); ++i)
            {
                shape[i + 2] = output_shape[i];
            }
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
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 3); // Remove {G,O,I}
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
    }
    else
    {
        if (filters_pshape.is_static() && data_pshape.is_static())
        {
            auto filters_shape = filters_pshape.to_shape();
            auto data_shape = data_pshape.to_shape();

            Shape output_shape;
            auto data_spatial_rank = data_shape.size() - 2;
            auto output_padding = get_output_padding();
            if (output_padding.size() == 0)
            {
                output_padding.insert(output_padding.begin(), data_spatial_rank, 0);
            }
            for (size_t i = 0; i < data_spatial_rank; ++i)
            {
                size_t tmp = m_strides[i] * (data_shape[i + 2] - 1) +
                             ((filters_shape[i + 3] - 1) * m_dilations[i] + 1) - m_pads_begin[i] -
                             m_pads_end[i] + output_padding[i];
                output_shape.push_back(tmp);
            }
            output_shape.insert(output_shape.begin(),
                                filters_shape.at(0) * filters_shape.at(2)); // GROUP * C_OUTPUT
            output_shape.insert(output_shape.begin(), data_shape.at(0));
            output_pshape = output_shape;
        }
    }

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, result_et, output_pshape);
}

void op::v1::GroupConvolutionBackpropData::generate_adjoints(autodiff::Adjoints& adjoints,
                                                             const OutputVector& deltas)
{
    ngraph_error("Not Yet Implemented");
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

constexpr NodeTypeInfo op::v0::GroupConvolution::type_info;

op::v0::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
                                           const Output<Node>& filters,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const size_t groups,
                                           const PadType& pad_type)
    : FusedOp({data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_groups(groups)
    , m_pad_type(pad_type)
    , m_groups_in_filters(false)
{
    constructor_validate_and_infer_types();
}

op::v0::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
                                           const Output<Node>& filters,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const PadType& pad_type)
    : FusedOp({data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_groups(0)
    , m_pad_type(pad_type)
    , m_groups_in_filters(true)
{
    constructor_validate_and_infer_types();
}

void op::v0::GroupConvolution::pre_validate_and_infer_types()
{
    auto data_shape = get_input_partial_shape(0);
    auto filters_shape = get_input_partial_shape(1);

    if (data_shape.is_static() && filters_shape.is_static())
    {
        // Update groups
        if (m_groups_in_filters)
        {
            m_groups = static_cast<size_t>(get_input_partial_shape(1)[0]);
        }

        // Data channels
        NODE_VALIDATION_CHECK(this,
                              data_shape.to_shape()[1] % get_groups() == 0,
                              "Data channels not a multiple of group size");
        // Output channels
        NODE_VALIDATION_CHECK(this,
                              filters_shape.to_shape()[0] % get_groups() == 0,
                              "# Filters not a multiple of group size");

        // Input Filters
        NODE_VALIDATION_CHECK(this,
                              (filters_shape.to_shape()[m_groups_in_filters ? 2 : 1] *
                               get_groups()) == data_shape.to_shape()[1],
                              "Incorrect number of channels per filter");
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

void op::v0::GroupConvolution::post_validate_and_infer_types()
{
    auto data_shape = get_input_partial_shape(0);
    auto filters_shape = get_input_partial_shape(1);
    if (data_shape.is_static() && filters_shape.is_static())
    {
        if (m_pad_type == PadType::SAME_UPPER || m_pad_type == PadType::SAME_LOWER)
        {
            m_padding_below.clear();
            m_padding_above.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            infer_auto_padding(data_shape.to_shape(),
                               filter_shape,
                               m_window_movement_strides,
                               m_window_dilation_strides,
                               m_pad_type,
                               m_padding_above,
                               m_padding_below);
        }
    }
}

Shape op::v0::GroupConvolution::get_weights_dimensions() const
{
    auto data_shape = get_input_shape(0);
    auto weights_shape = get_input_shape(1);
    // check if weights already includes groups
    if (m_groups_in_filters)
    {
        return weights_shape;
    }
    // reshape weights into 5d tensors that includes groups
    const size_t OC = 0;
    const size_t OC_IN_OUTPUT = 1;
    const size_t IC = 1;
    Shape weights_shape_groups{weights_shape};
    // adjust output and channel given a number of groups

    weights_shape_groups.at(OC) = get_shape().at(OC_IN_OUTPUT) / get_groups();
    weights_shape_groups.at(IC) = data_shape.at(IC) / get_groups();
    // push_front the number of groups
    weights_shape_groups.insert(weights_shape_groups.begin(), get_groups());
    return weights_shape_groups;
}

shared_ptr<Node> op::v0::GroupConvolution::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);

    if (m_groups_in_filters)
    {
        return make_shared<op::v0::GroupConvolution>(new_args.at(0),
                                                     new_args.at(1),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     get_pad_type());
    }
    else
    {
        return make_shared<op::v0::GroupConvolution>(new_args.at(0),
                                                     new_args.at(1),
                                                     get_window_movement_strides(),
                                                     get_window_dilation_strides(),
                                                     get_padding_below(),
                                                     get_padding_above(),
                                                     get_data_dilation_strides(),
                                                     get_groups(),
                                                     get_pad_type());
    }
}

NodeVector op::v0::GroupConvolution::decompose_op() const
{
    auto data = input_value(0);
    auto filters = input_value(1);
    auto filters_shape = get_input_shape(1);
    // Split one convolution op to N ops where N is the number of groups
    // and concat results after computation.
    // reference:
    // https://github.com/NervanaSystems/ngraph-mxnet/blob/fdd692/src/ngraph/ngraph_emitter.cc#L822-L856
    NodeVector convolution_nodes;

    // slice data
    auto sliced_data = builder::split(data, get_groups(), 1);
    // slice filters
    auto sliced_filters = builder::split(filters, get_groups(), 0);
    for (std::size_t group{0}; group < get_groups(); ++group)
    {
        auto sliced_filter = sliced_filters[group];
        if (m_groups_in_filters)
        {
            // Remove group dimmension after slicing
            sliced_filter = make_shared<op::Reshape>(
                sliced_filters[group],
                get_default_order(sliced_filters[group]->get_shape().size()),
                Shape(std::next(std::begin(filters_shape), 1), std::end(filters_shape)));
        }
        convolution_nodes.push_back(
            std::make_shared<ngraph::op::Convolution>(sliced_data[group],
                                                      sliced_filter,
                                                      m_window_movement_strides,
                                                      m_window_dilation_strides,
                                                      m_padding_below,
                                                      m_padding_above,
                                                      m_data_dilation_strides,
                                                      m_pad_type));
    }
    std::size_t concatenation_axis = 1;
    return {std::make_shared<ngraph::op::Concat>(convolution_nodes, concatenation_axis)};
}

void op::GroupConvolution::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                             const OutputVector& /* deltas */)
{
    throw ngraph_error("NYI");
}

constexpr NodeTypeInfo op::v0::GroupConvolutionBackpropData::type_info;

op::v0::GroupConvolutionBackpropData::GroupConvolutionBackpropData(
    const Output<Node>& data_batch,
    const Output<Node>& filters,
    const Output<Node>& output_delta,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above,
    const size_t groups)
    : FusedOp({data_batch, filters, output_delta})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_groups(groups)
{
    constructor_validate_and_infer_types();
}

void op::v0::GroupConvolutionBackpropData::pre_validate_and_infer_types()
{
    element::Type data_element_type = get_input_element_type(0);
    PartialShape data_pshape = get_input_partial_shape(0);
    PartialShape filters_pshape = get_input_partial_shape(1);
    PartialShape delta_pshape = get_input_partial_shape(2);

    NODE_VALIDATION_CHECK(this,
                          data_element_type.is_dynamic() || data_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          data_element_type,
                          ").");

    if (data_pshape.is_dynamic() || filters_pshape.is_dynamic() || delta_pshape.is_dynamic())
    {
        set_output_type(0, data_element_type, PartialShape::dynamic());
    }
}

shared_ptr<Node>
    op::v0::GroupConvolutionBackpropData::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<op::v0::GroupConvolutionBackpropData>(new_args.at(0),
                                                             new_args.at(1),
                                                             new_args.at(2),
                                                             get_window_movement_strides(),
                                                             get_window_dilation_strides(),
                                                             get_padding_below(),
                                                             get_padding_above(),
                                                             get_groups());
}

NodeVector op::v0::GroupConvolutionBackpropData::decompose_op() const
{
    auto filters = input_value(1);
    auto output_delta = input_value(2);

    auto data_shape = get_input_shape(0);

    NodeVector sliced_inputs;

    auto groups = get_groups();
    // slice data shape
    data_shape[1] /= groups;
    // slice delta
    auto sliced_delta = builder::split(output_delta, groups, 1);
    // slice filters
    auto sliced_filters = builder::split(filters, groups, 0);

    auto num_spatials = get_window_movement_strides().size();

    for (size_t i = 0; i < groups; ++i)
    {
        auto sliced_conv = std::make_shared<op::ConvolutionBackpropData>(
            data_shape,
            sliced_filters[i],
            sliced_delta[i],
            get_window_movement_strides(),
            get_window_dilation_strides(),
            get_padding_below(),
            get_padding_above(),
            Strides(num_spatials, 1)); // default data dilation strides

        sliced_inputs.push_back(sliced_conv);
    }

    size_t concatenation_axis = 1;
    return {std::make_shared<ngraph::op::Concat>(sliced_inputs, concatenation_axis)};
}

constexpr NodeTypeInfo op::v0::GroupConvolutionBackpropFilters::type_info;

op::v0::GroupConvolutionBackpropFilters::GroupConvolutionBackpropFilters(
    const Output<Node>& data_batch,
    const Output<Node>& filters,
    const Output<Node>& output_delta,
    const Strides& window_movement_strides,
    const Strides& window_dilation_strides,
    const CoordinateDiff& padding_below,
    const CoordinateDiff& padding_above,
    const size_t groups)
    : FusedOp({data_batch, filters, output_delta})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_groups(groups)
{
    constructor_validate_and_infer_types();
}

void op::v0::GroupConvolutionBackpropFilters::pre_validate_and_infer_types()
{
    element::Type filters_element_type = get_input_element_type(1);
    PartialShape data_pshape = get_input_partial_shape(0);
    PartialShape filters_pshape = get_input_partial_shape(1);
    PartialShape delta_pshape = get_input_partial_shape(2);

    NODE_VALIDATION_CHECK(this,
                          filters_element_type.is_dynamic() || filters_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          filters_element_type,
                          ").");

    if (data_pshape.is_dynamic() || filters_pshape.is_dynamic() || delta_pshape.is_dynamic())
    {
        set_output_type(0, filters_element_type, PartialShape::dynamic());
    }
}

shared_ptr<Node>
    op::v0::GroupConvolutionBackpropFilters::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<op::v0::GroupConvolutionBackpropFilters>(new_args.at(0),
                                                                new_args.at(1),
                                                                new_args.at(2),
                                                                get_window_movement_strides(),
                                                                get_window_dilation_strides(),
                                                                get_padding_below(),
                                                                get_padding_above(),
                                                                get_groups());
}

NodeVector op::v0::GroupConvolutionBackpropFilters::decompose_op() const
{
    auto data_batch = input_value(0);
    auto filters = input_value(1);
    auto output_delta = input_value(2);

    auto data_shape = get_input_shape(0);
    auto filters_shape = get_input_shape(1);
    auto delta_shape = get_input_shape(2);

    NodeVector sliced_inputs;

    for (size_t i = 0; i < get_groups(); ++i)
    {
        size_t channel_step = filters_shape.at(1);

        const Coordinate data_lower_bound{0, i * channel_step, 0, 0};
        const Coordinate data_upper_bound{
            data_shape.at(0), (i + 1) * channel_step, data_shape.at(2), data_shape.at(3)};
        auto sliced_data =
            std::make_shared<op::Slice>(data_batch, data_lower_bound, data_upper_bound);

        size_t filters_step = filters_shape.at(0) / get_groups();

        const Coordinate filters_lower_bound{i * filters_step, 0, 0, 0};
        const Coordinate filters_upper_bound{
            (i + 1) * filters_step, filters_shape.at(1), filters_shape.at(2), filters_shape.at(3)};
        auto sliced_filters =
            std::make_shared<op::Slice>(filters, filters_lower_bound, filters_upper_bound);

        const Coordinate delta_lower_bound{0, i * filters_step, 0, 0};
        const Coordinate delta_upper_bound{
            delta_shape.at(0), (i + 1) * filters_step, delta_shape.at(2), delta_shape.at(3)};
        auto sliced_delta =
            std::make_shared<op::Slice>(output_delta, delta_lower_bound, delta_upper_bound);

        auto sliced_conv =
            std::make_shared<op::ConvolutionBackpropFilters>(sliced_data,
                                                             sliced_filters->get_shape(),
                                                             sliced_delta,
                                                             get_window_movement_strides(),
                                                             get_window_dilation_strides(),
                                                             get_padding_below(),
                                                             get_padding_above(),
                                                             Strides{1, 1});

        sliced_inputs.push_back(sliced_conv);
    }

    size_t concatenation_axis = 0;
    return {std::make_shared<ngraph::op::Concat>(sliced_inputs, concatenation_axis)};
}
