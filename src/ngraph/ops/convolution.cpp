// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ops/convolution.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::Convolution::Convolution(const std::shared_ptr<Node>& arg0,
                             const std::shared_ptr<Node>& arg1,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides)
    : RequiresTensorViewArgs("Convolution", {arg0,arg1})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
{
    auto arg0_tensor_view_type = get_inputs().at(0).get_tensor_view_type();
    auto& arg0_shape = arg0_tensor_view_type->get_shape();

    auto arg1_tensor_view_type = get_inputs().at(1).get_tensor_view_type();
    auto& arg1_shape = arg1_tensor_view_type->get_shape();

    //
    // Make sure arg0: NCiDi for some Di of rank>0, N != 0, Ci != 0.
    //
    if (arg0_shape.size() < 3)
    {
        throw ngraph_error("Convolution image batch input must have rank of at least 3 (one batch axis, one input-channel axis, at least one image dimension).");
    }

    m_batch_size = arg0_shape[0];
    if (m_batch_size == 0)
    {
        throw ngraph_error("Convolution image batch size is zero.");
    }

    m_n_input_channels = arg0_shape[1];
    if (m_n_input_channels == 0)
    {
        throw ngraph_error("Convolution requires at least one input channel.");
    }

    m_n_image_dimensions = arg0_shape.size() - 2;

    //
    // Make sure arg1: CoCiWv for some Co>0, rank of W = rank of Di.
    //
    if (arg1_shape.size() != 2 + m_n_image_dimensions)
    {
        throw ngraph_error("Convolution filter input must have rank of 2 + n_image_dimensions.");
    }

    m_n_output_channels = arg1_shape[0];
    if (m_n_output_channels == 0)
    {
        throw ngraph_error("Convolution requires at least one output channel.");
    }

    if (arg1_shape[1] != m_n_input_channels)
    {
        throw ngraph_error("Convolution image batch and filter input channel counts do not match.");
    }

    //
    // Make sure window movement strides and window dilation strades have same rank as Di.
    //
    if (m_window_movement_strides.size() != m_n_image_dimensions)
    {
        throw ngraph_error("Convolution window movement stride rank does not match number of image dimensions.");
    }

    if (m_window_dilation_strides.size() != m_n_image_dimensions)
    {
        throw ngraph_error("Convolution window dilation stride rank does not match number of image dimensions.");
    }

    //
    // Extract input image shape Di and make sure all dimensions are larger than 0.
    //
    for (size_t i = 0; i < m_n_image_dimensions; i++)
    {
        m_input_image_shape.push_back(arg0_shape[1 + 1 + +i]);

        if (m_input_image_shape[i] == 0)
        {
            throw ngraph_error("Convolution input image dimension is zero.");
        }
    }

    //
    // Extract the virtual shape Wv of the convolution window, *not* including dilation, from the filter dimensions.
    // At the same time, make sure window shape dimensions are all larger than 0.
    //
    for (size_t i = 0; i < m_n_image_dimensions; i++)
    {
        m_window_virtual_shape.push_back(arg1_shape[1 + 1 + i]);
        if (m_window_virtual_shape[i] == 0)
        {
            throw ngraph_error("Convolution window shape has a zero-length axis.");
        }
    }

    //
    // Compute physical shape Wp of the convolution window, *including* dilation. At the same time, make sure all 
    // window dilation strides are larger than 0, and that the dilated filter fits within the image dimensions.
    //
    for (size_t i = 0; i < m_n_image_dimensions; i++)
    {
        if (m_window_dilation_strides[i] == 0)
        {
            throw ngraph_error("Convolution window axis stride is zero.");
        }

        m_window_physical_shape.push_back((m_window_virtual_shape[i] - 1) * m_window_dilation_strides[i] + 1);

        if (m_window_physical_shape[i] > m_input_image_shape[i])
        {
            throw ngraph_error("Convolution window after dilation is larger than the image.");
        }
    }

    //
    // Compute image output shape Do, checking at the same time that all window movement strides are larger than 0.
    //
    for (size_t i = 0; i < m_n_image_dimensions; i++)
    {
        if (m_window_movement_strides[i] == 0)
        {
            throw ngraph_error("Convolution window movement stride is zero.");
        }
        m_output_image_shape.push_back(ceil_div(m_input_image_shape[i] - m_window_physical_shape[i] + 1,m_window_movement_strides[i]));
    }

    //
    // Construct result shape: NCoDo.
    //
    Shape result_shape(1 + 1 + m_n_image_dimensions);
    result_shape[0] = m_batch_size;
    result_shape[1] = m_n_output_channels;
    std::copy(m_output_image_shape.begin(), m_output_image_shape.end(), result_shape.begin() + 2);

    set_value_type_checked(
        make_shared<TensorViewType>(arg0_tensor_view_type->get_element_type(), result_shape));
}

Strides default_strides(const std::shared_ptr<Node>& arg0)
{
    auto arg0_value_type = arg0->get_value_type();
    auto arg0_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg0_value_type);
    if (arg0_tensor_view_type == nullptr)
    {
        throw ngraph_error("Convolution image batch argument has non-tensor view type");
    }
    auto& arg0_shape = arg0_tensor_view_type->get_shape();
    if (arg0_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error("Convolution image batch input must have rank of at least 3 (one batch axis, one input-channel axis, at least one image dimension).");
    }
    return Strides(arg0_shape.size() - 2,1);
}

op::Convolution::Convolution(const std::shared_ptr<Node>& arg0,
                             const std::shared_ptr<Node>& arg1,
                             const Strides& window_movement_strides)
    : Convolution(arg0,arg1,window_movement_strides,default_strides(arg0))
{
}

op::Convolution::Convolution(const std::shared_ptr<Node>& arg0,
                             const std::shared_ptr<Node>& arg1)
    : Convolution(arg0,arg1,default_strides(arg0),default_strides(arg0))
{
}

/*
void op::Convolution::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
}
*/
