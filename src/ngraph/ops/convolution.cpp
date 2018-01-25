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

#include <numeric>

#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static Shape infer_convolution_output_shape(const Shape& image_batch_shape,
                                            const Shape& filters_shape,
                                            const Strides& window_movement_strides,
                                            const Strides& window_dilation_strides,
                                            const CoordinateDiff& padding_below,
                                            const CoordinateDiff& padding_above,
                                            const Strides& image_dilation_strides)
{
    //
    // Make sure image_batch: NCiDi for some Di of rank>0, N != 0, Ci != 0.
    //
    if (image_batch_shape.size() < 3)
    {
        throw ngraph_error(
            "Convolution image batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one image dimension).");
    }

    size_t batch_size = image_batch_shape[0];
    if (batch_size == 0)
    {
        throw ngraph_error("Convolution image batch size is zero.");
    }

    size_t input_channel_count = image_batch_shape[1];
    if (input_channel_count == 0)
    {
        throw ngraph_error("Convolution requires at least one input channel.");
    }

    size_t image_dimension_count = image_batch_shape.size() - 2;

    //
    // Make sure filters: CoCiWv for some Co>0, rank of W = rank of Di.
    //
    if (filters_shape.size() != 2 + image_dimension_count)
    {
        throw ngraph_error("Convolution filter input must have rank of 2 + n_image_dimensions.");
    }

    size_t output_channel_count = filters_shape[0];
    if (output_channel_count == 0)
    {
        throw ngraph_error("Convolution requires at least one output channel.");
    }

    if (filters_shape[1] != input_channel_count)
    {
        throw ngraph_error("Convolution image batch and filter input channel counts do not match.");
    }

    //
    // Make sure window movement strides, window dilation strides, and image dilation strides
    // have same rank as Di.
    //
    if (window_movement_strides.size() != image_dimension_count)
    {
        throw ngraph_error(
            "Convolution window movement stride rank does not match number of image dimensions.");
    }

    if (window_dilation_strides.size() != image_dimension_count)
    {
        throw ngraph_error(
            "Convolution window dilation stride rank does not match number of image dimensions.");
    }

    if (image_dilation_strides.size() != image_dimension_count)
    {
        throw ngraph_error(
            "Convolution image dilation stride rank does not match number of image dimensions.");
    }

    //
    // Make sure padding-below and padding-above shapes have same rank as Di.
    //
    if (padding_below.size() != image_dimension_count)
    {
        throw ngraph_error(
            "Convolution padding-below rank does not match number of image dimensions.");
    }

    if (padding_above.size() != image_dimension_count)
    {
        throw ngraph_error(
            "Convolution padding-above rank does not match number of image dimensions.");
    }

    //
    // Extract input image shape Di and make sure all dimensions are larger than 0 after padding and dilation.
    //
    Shape input_image_virtual_shape;

    for (size_t i = 0; i < image_dimension_count; i++)
    {
        if (image_dilation_strides[i] == 0)
        {
            throw ngraph_error("Convolution image dilation stride is zero.");
        }

        size_t dim_size = image_batch_shape[1 + 1 + i];
        size_t dilated_dim_size = (dim_size - 1) * image_dilation_strides[i] + 1;

        std::ptrdiff_t padded_dilated_dim_size =
            padding_below[i] + dilated_dim_size + padding_above[i];

        if (padded_dilated_dim_size < 0)
        {
            throw ngraph_error(
                "Convolution input image dimension after padding and dilation is negative.");
        }

        input_image_virtual_shape.push_back(padded_dilated_dim_size);

        if (input_image_virtual_shape[i] == 0)
        {
            throw ngraph_error(
                "Convolution input image dimension after dilation is zero even with padding.");
        }
    }

    //
    // Extract the physical shape Wp of the convolution window, *not* including dilation, from the filter dimensions.
    // At the same time, make sure window shape dimensions are all larger than 0.
    //
    Shape window_physical_shape;

    for (size_t i = 0; i < image_dimension_count; i++)
    {
        window_physical_shape.push_back(filters_shape[1 + 1 + i]);
        if (window_physical_shape[i] == 0)
        {
            throw ngraph_error("Convolution window shape has a zero-length axis.");
        }
    }

    //
    // Compute physical shape Wp of the convolution window, *including* dilation. At the same time, make sure all
    // window dilation strides are larger than 0, and that the dilated filter fits within the image dimensions.
    //
    Shape window_virtual_shape;

    for (size_t i = 0; i < image_dimension_count; i++)
    {
        if (window_dilation_strides[i] == 0)
        {
            throw ngraph_error("Convolution window axis dilation stride is zero.");
        }

        window_virtual_shape.push_back((window_physical_shape[i] - 1) * window_dilation_strides[i] +
                                       1);

        if (window_virtual_shape[i] > input_image_virtual_shape[i])
        {
            throw ngraph_error(
                "Convolution window after dilation is larger than the image even with padding.");
        }
    }

    //
    // Construct result shape: NCoDo, checking at the same time that all window movement strides are larger than 0.
    //
    Shape result_shape;
    result_shape.push_back(batch_size);
    result_shape.push_back(output_channel_count);

    for (size_t i = 0; i < image_dimension_count; i++)
    {
        if (window_movement_strides[i] == 0)
        {
            throw ngraph_error("Convolution window axis movement stride is zero.");
        }
        result_shape.push_back(ceil_div(input_image_virtual_shape[i] - window_virtual_shape[i] + 1,
                                        window_movement_strides[i]));
    }

    return result_shape;
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const CoordinateDiff& padding_below,
                             const CoordinateDiff& padding_above,
                             const Strides& image_dilation_strides)
    : RequiresTensorViewArgs("Convolution", {image_batch, filters})
    , m_window_movement_strides(window_movement_strides)
    , m_window_dilation_strides(window_dilation_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_image_dilation_strides(image_dilation_strides)
{
    auto& image_batch_shape = get_inputs().at(0).get_shape();
    auto& image_batch_et = get_inputs().at(0).get_element_type();
    auto& filters_shape = get_inputs().at(1).get_shape();
    auto& filters_et = get_inputs().at(1).get_element_type();

    //
    // Make sure image batch and filter element types match.
    //
    if (image_batch_et != filters_et)
    {
        throw ngraph_error("Convolution image batch and filter element types do not match");
    }

    set_value_type_checked(image_batch_et,
                           infer_convolution_output_shape(image_batch_shape,
                                                          filters_shape,
                                                          window_movement_strides,
                                                          window_dilation_strides,
                                                          padding_below,
                                                          padding_above,
                                                          image_dilation_strides));
}

Strides op::Convolution::default_strides(const std::shared_ptr<Node>& image_batch)
{
    auto& image_batch_shape = image_batch->get_shape();
    if (image_batch_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Convolution image batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one image dimension).");
    }
    return Strides(image_batch_shape.size() - 2, 1);
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const CoordinateDiff& padding_below,
                             const CoordinateDiff& padding_above)
    : Convolution(image_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  padding_below,
                  padding_above,
                  default_strides(image_batch))
{
}

CoordinateDiff op::Convolution::default_padding(const std::shared_ptr<Node>& image_batch)
{
    auto& image_batch_shape = image_batch->get_shape();
    if (image_batch_shape.size() < 3)
    {
        // For consistency we should throw the same error message here that we throw in the constructor.
        throw ngraph_error(
            "Convolution image batch input must have rank of at least 3 (one batch axis, one "
            "input-channel axis, at least one image dimension).");
    }
    return CoordinateDiff(image_batch_shape.size() - 2, 0);
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides)
    : Convolution(image_batch,
                  filters,
                  window_movement_strides,
                  window_dilation_strides,
                  default_padding(image_batch),
                  default_padding(image_batch))
{
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters,
                             const Strides& window_movement_strides)
    : Convolution(image_batch,
                  filters,
                  window_movement_strides,
                  default_strides(image_batch),
                  default_padding(image_batch),
                  default_padding(image_batch))
{
}

op::Convolution::Convolution(const std::shared_ptr<Node>& image_batch,
                             const std::shared_ptr<Node>& filters)
    : Convolution(image_batch,
                  filters,
                  default_strides(image_batch),
                  default_strides(image_batch),
                  default_padding(image_batch),
                  default_padding(image_batch))
{
}

std::shared_ptr<Node>
    op::Convolution::copy_with_new_args(const std::vector<std::shared_ptr<Node>>& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<Convolution>(new_args.at(0),
                                         new_args.at(1),
                                         m_window_movement_strides,
                                         m_window_dilation_strides,
                                         m_padding_below,
                                         m_padding_above,
                                         m_image_dilation_strides);
}

bool op::Convolution::is_functionally_identical(const Node& other) const
{
    bool rc = true;
    if (Node::test_identical(other))
    {
        const Convolution& rhs = dynamic_cast<const Convolution&>(other);
        rc &= m_window_movement_strides == rhs.m_window_movement_strides;
        rc &= m_window_dilation_strides == rhs.m_window_dilation_strides;
        rc &= m_padding_below == rhs.m_padding_below;
        rc &= m_padding_above == rhs.m_padding_above;
        rc &= m_image_dilation_strides == rhs.m_image_dilation_strides;
    }
    else
    {
        rc = false;
    }
    return rc;
}

std::shared_ptr<op::Reshape> flipDim0and1(std::shared_ptr<Node> node, const Shape& shape)
{
    AxisVector input_order(shape.size());
    std::iota(input_order.begin(), input_order.end(), 0);
    std::swap(input_order[0], input_order[1]);

    auto output_shape = shape;
    std::swap(output_shape[0], output_shape[1]);

    return std::make_shared<op::Reshape>(node, input_order, output_shape);
}

void op::Convolution::generate_adjoints(autodiff::Adjoints& adjoints,
                                        const std::shared_ptr<Node>& delta)
{
    // input
    // {N, Cin, d1,...,dn}
    auto x = get_input_op(0);
    const auto x_shape = x->get_shape();

    // filters
    // {Cout, Cin, df1,...,dfn}
    auto f = get_input_op(1);
    const auto f_shape = f->get_shape();

    // {N, Cout, d'1,...,d'n}
    const auto delta_shape = delta->get_shape();

    AxisSet data_axes_2_to_n;

    // adjust padding for x and f adjoints per:
    // https://wiki.ith.intel.com/display/intelnervana/Autodiff
    CoordinateDiff x_adjoint_padding_below;
    CoordinateDiff x_adjoint_padding_above;
    CoordinateDiff f_adjoint_padding_below;
    CoordinateDiff f_adjoint_padding_above;

    // loop over data axes
    for (auto i = 0; i < delta_shape.size() - 2; ++i)
    {
        data_axes_2_to_n.insert(i + 2);

        // (Sw - 1)%Q
        // (Ax + (Sx - 1)Px + Bx - (Sf - 1)Pf) % Q
        auto sw_mod_q = (m_padding_below[i] + (x_shape[i + 2] - 1) * m_image_dilation_strides[i] +
                         m_padding_above[i] - (f_shape[i + 2] - 1) * m_window_dilation_strides[i]) %
                        m_window_movement_strides[i];

        // (Sf - 1)Pf + (Sw - 1)%Q - Bx
        x_adjoint_padding_above.push_back((f_shape[i + 2] - 1) * m_window_dilation_strides[i] +
                                          sw_mod_q - m_padding_above[i]);

        // (Sf - 1)Pf - Ax
        x_adjoint_padding_below.push_back((f_shape[i + 2] - 1) * m_window_dilation_strides[i] -
                                          m_padding_below[i]);

        // Bx  - (SW - 1)%Q
        f_adjoint_padding_above.push_back(m_padding_above[i] - sw_mod_q);

        // Ax
        f_adjoint_padding_below.push_back(m_padding_below[i]);
    }

    // to calculate adjoint of the input...
    // 1) reshape filter (flip channel dimensions)
    // {Cin, Cout, df1,...,dfn}
    auto f_reshape = flipDim0and1(f, f_shape);

    // 2) reverse filter data
    auto f_reshape_reverse = std::make_shared<op::Reverse>(f_reshape, data_axes_2_to_n);

    // 3) convolve delta with reshaped/reversed filter
    //    swap image_dilation_stride and window_movement_stride
    // {N, Cin, d1,...,dn}
    auto x_adjoint = std::make_shared<op::Convolution>(delta,
                                                       f_reshape_reverse,
                                                       m_image_dilation_strides,
                                                       m_window_dilation_strides,
                                                       x_adjoint_padding_below,
                                                       x_adjoint_padding_above,
                                                       m_window_movement_strides);

    adjoints.add_delta(x, x_adjoint);

    // to calculate adjoint of the filter...
    // 1) reshape input
    // {Cin, N, d1,...,dn}
    auto x_reshape = flipDim0and1(x, x_shape);

    // 2) reshape delta
    // {Cout, N, d'1,...d'n}
    auto delta_reshape = flipDim0and1(delta, delta_shape);

    // 3) convolve reshaped input with reshaped delta
    //    swap window_movement_stride and window_dilation_stride
    // {Cin, Cout, df1,...,dfn}
    auto f_adjoint = std::make_shared<op::Convolution>(x_reshape,
                                                       delta_reshape,
                                                       m_window_dilation_strides,
                                                       m_window_movement_strides,
                                                       f_adjoint_padding_below,
                                                       f_adjoint_padding_above,
                                                       m_image_dilation_strides);

    // 4) reshape result to match filter dimentions
    // {Cout, Cin, df1,...,dfn}
    adjoints.add_delta(f, flipDim0and1(f_adjoint, f_adjoint->get_shape()));
}
