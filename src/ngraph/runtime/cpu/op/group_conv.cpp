//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::GroupConvolution::GroupConvolution(const shared_ptr<Node>& data_batch,
                                       const shared_ptr<Node>& filters,
                                       const Strides& window_movement_strides,
                                       const Strides& window_dilation_strides,
                                       const CoordinateDiff& padding_below,
                                       const CoordinateDiff& padding_above,
                                       const Strides& data_dilation_strides,
                                       size_t groups,
                                       const Shape& output_shape)
    : RequiresTensorViewArgs("GroupConvolution", {data_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_data_dilation_strides(data_dilation_strides)
    , m_groups(groups)
{
    auto& data_batch_et = data_batch->get_element_type();
    auto& filters_et = filters->get_element_type();

    //
    // Make sure data batch and filter element types match.
    //
    if (data_batch_et != filters_et)
    {
        throw ngraph_error("Convolution data batch and filter element types do not match");
    }

    set_value_type_checked(data_batch_et, output_shape);
}

Shape op::GroupConvolution::get_weights_dimensions() const
{
    // reshape weights into 5d tensors that includes groups
    const size_t OC = 0;
    const size_t OC_IN_OUTPUT = 1;
    const size_t IC = 1;
    Shape weights_shape_groups{get_inputs().at(1).get_shape()};
    // adjust output and channel given a number of groups

    weights_shape_groups.at(OC) = get_shape().at(OC_IN_OUTPUT) / get_groups();
    weights_shape_groups.at(IC) = get_inputs().at(0).get_shape().at(IC) / get_groups();
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
                                             this->get_shape());
}

void op::GroupConvolution::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("NYI");
}
