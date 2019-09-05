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

#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

const string op::GroupConvolution::type_name{"GroupConvolution"};

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

void op::GroupConvolution::pre_validate_and_infer_types()
{
    auto data_shape = get_input_partial_shape(0);
    auto filters_shape = get_input_partial_shape(1);
    if (data_shape.is_static() && filters_shape.is_static())
    {
        // Data channels
        NODE_VALIDATION_CHECK(this,
                              data_shape.to_shape()[1] % m_groups == 0,
                              "Data channels not a multiple of group size");
        // Output channels
        NODE_VALIDATION_CHECK(this,
                              filters_shape.to_shape()[0] % m_groups == 0,
                              "# Filters not a multiple of group size");
        // Input Filters
        NODE_VALIDATION_CHECK(this,
                              filters_shape.to_shape()[1] * m_groups == data_shape.to_shape()[1],
                              "Incorrect number of channels per filter");
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
    // reshape weights into 5d tensors that includes groups
    const size_t OC = 0;
    const size_t OC_IN_OUTPUT = 1;
    const size_t IC = 1;
    Shape weights_shape_groups{get_input_shape(1)};
    // adjust output and channel given a number of groups

    weights_shape_groups.at(OC) = get_shape().at(OC_IN_OUTPUT) / get_groups();
    weights_shape_groups.at(IC) = get_input_shape(0).at(IC) / get_groups();
    // push_front the number of groups
    weights_shape_groups.insert(weights_shape_groups.begin(), get_groups());
    return weights_shape_groups;
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
    // Split one convolution op to N ops where N is the number of groups
    // and concat results after computation.
    // reference:
    // https://github.com/NervanaSystems/ngraph-mxnet/blob/fdd692/src/ngraph/ngraph_emitter.cc#L822-L856
    std::size_t n_data_channels{data.get_shape().at(1)};
    std::size_t n_filters_channels{filters.get_shape().at(0)};
    std::size_t data_group_size{n_data_channels / m_groups};
    std::size_t filters_group_size{n_filters_channels / m_groups};
    NodeVector convolution_nodes;

    // initial bounds for splice
    std::vector<std::size_t> data_lower_bounds(data.get_shape().size());
    std::vector<std::size_t> data_upper_bounds{data.get_shape()};
    std::vector<std::size_t> filters_lower_bounds(filters.get_shape().size());
    std::vector<std::size_t> filters_upper_bounds{filters.get_shape()};

    for (std::size_t group{0}; group < m_groups; ++group)
    {
        // slice data
        data_lower_bounds[1] = group * data_group_size;
        data_upper_bounds[1] = (group + 1) * data_group_size;
        auto sliced_data =
            std::make_shared<ngraph::op::Slice>(data, data_lower_bounds, data_upper_bounds);
        // slice filters
        filters_lower_bounds[0] = group * filters_group_size;
        filters_upper_bounds[0] = (group + 1) * filters_group_size;
        auto sliced_filters = std::make_shared<ngraph::op::Slice>(
            filters, filters_lower_bounds, filters_upper_bounds);

        convolution_nodes.push_back(
            std::make_shared<ngraph::op::Convolution>(sliced_data,
                                                      sliced_filters,
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

void op::GroupConvolution::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("NYI");
}
