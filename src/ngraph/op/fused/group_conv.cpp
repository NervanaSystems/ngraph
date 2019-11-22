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

#include <numeric>

#include "group_conv.hpp"

#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GroupConvolution::type_info;

op::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
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
{
    constructor_validate_and_infer_types();
}

op::GroupConvolution::GroupConvolution(const Output<Node>& data_batch,
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
    , m_groups(filters.get_partial_shape().rank().is_dynamic() ? Dimension::dynamic()
                                                               : filters.get_partial_shape()[0])
    , m_pad_type(pad_type)
{
    constructor_validate_and_infer_types();
}

void op::GroupConvolution::pre_validate_and_infer_types()
{
    auto data_shape = get_input_partial_shape(0);
    auto filters_shape = get_input_partial_shape(1);

    if (data_shape.is_static() && filters_shape.is_static())
    {
        // Update groups
        if (has_groups_in_filters_shape())
        {
            m_groups = get_input_partial_shape(1)[0];
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
                              (filters_shape.to_shape()[has_groups_in_filters_shape() ? 2 : 1] *
                               get_groups()) == data_shape.to_shape()[1],
                              "Incorrect number of channels per filter");
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

void op::GroupConvolution::post_validate_and_infer_types()
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

Shape op::GroupConvolution::get_weights_dimensions() const
{
    auto data_shape = get_input_shape(0);
    auto weights_shape = get_input_shape(1);
    // check if weights already includes groups
    if (has_groups_in_filters_shape())
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

size_t ngraph::op::GroupConvolution::get_groups() const
{
    NODE_VALIDATION_CHECK(this,
                          m_groups.is_static(),
                          "get_groups() can only be called if the number of groups is static.");
    return static_cast<size_t>(m_groups);
}

shared_ptr<Node> op::GroupConvolution::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<op::GroupConvolution>(new_args.at(0),
                                             new_args.at(1),
                                             get_window_movement_strides(),
                                             get_window_dilation_strides(),
                                             get_padding_below(),
                                             get_padding_above(),
                                             get_data_dilation_strides(),
                                             get_groups(),
                                             get_pad_type());
}

NodeVector op::GroupConvolution::decompose_op() const
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
        if (has_groups_in_filters_shape())
        {
            // Remove group dimmension after slicing
            sliced_filter = builder::reshape(
                sliced_filters[group],
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
                                             const NodeVector& /* deltas */)
{
    throw ngraph_error("NYI");
}

bool ngraph::op::GroupConvolution::has_groups_in_filters_shape() const
{
    // If filters_rank is (data_rank + 1), then filters are divided by groups on first
    // dim.
    return ((get_input_shape(0).size() + 1) == get_input_shape(1).size());
}
