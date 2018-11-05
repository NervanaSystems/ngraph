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

#include "ngraph/log.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/runtime/plaidml/plaidml_convpool_formatter.hpp"
#include "ngraph/runtime/plaidml/plaidml_impl.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            template <typename O>
            class ConvolutionImpl : public BaseImpl<O>
            {
            public:
                ConvolutionImpl(Build* build, const O& op)
                    : BaseImpl<O>{build, op}
                {
                }

                void LogConvolution(vertexai::plaidml::variable image,
                                    vertexai::plaidml::variable filter,
                                    std::size_t image_dims,
                                    const Strides& window_movement_strides,
                                    const Strides& window_dilation_strides,
                                    const CoordinateDiff& padding_below,
                                    const CoordinateDiff& padding_above,
                                    const Strides& data_dilation_strides,
                                    std::size_t batch_axis_data,
                                    std::size_t input_channel_axis_data,
                                    std::size_t input_channel_axis_filters,
                                    std::size_t output_channel_axis_filters,
                                    std::size_t batch_axis_result,
                                    std::size_t output_channel_axis_result,
                                    bool rotate_filter);
            };

            template <>
            struct ParentImpl<op::Convolution>
            {
                using Type = ConvolutionImpl<op::Convolution>;
            };

            template <>
            struct ParentImpl<op::ConvolutionBackpropFilters>
            {
                using Type = ConvolutionImpl<op::ConvolutionBackpropFilters>;
            };

            template <>
            struct ParentImpl<op::ConvolutionBackpropData>
            {
                using Type = ConvolutionImpl<op::ConvolutionBackpropData>;
            };

            // Convolution implements a standard ML convolultion, with optional striding, padding, and dilation.
            template <>
            void Impl<op::Convolution>::operator()()
            {
                this->check_inputs(2);
                this->check_outputs(1);

                LogConvolution(op_input(0),
                               op_input(1),
                               op().get_inputs()[0].get_shape().size() - 2,
                               op().get_window_movement_strides(),
                               op().get_window_dilation_strides(),
                               op().get_padding_below(),
                               op().get_padding_above(),
                               op().get_data_dilation_strides(),
                               0,
                               1,
                               1,
                               0,
                               0,
                               1,
                               false);

                const auto& image = op_input(0);
                const auto& filter = op_input(1);
                auto image_dims = op().get_inputs()[0].get_shape().size() - 2;
                const auto& padding_above = op().get_padding_above();
                const auto& padding_below = op().get_padding_below();
                const auto& strides = op().get_window_movement_strides();
                const auto& filter_dilation = op().get_window_dilation_strides();
                const auto& data_dilation = op().get_data_dilation_strides();

                ConvPoolFormatter cpf(image_dims,
                                      padding_below,
                                      padding_above,
                                      strides,
                                      filter_dilation,
                                      data_dilation,
                                      ConvPoolFormatter::OpType::Conv,
                                      ConvPoolFormatter::DerivType::None);

                this->set_output(start_tile_function()
                                     .add(cpf.I_in_header(image))
                                     .add(cpf.F_in_header(filter))
                                     .add(cpf.O_out_header())
                                     .add(builder::BinaryContraction{"+", "*"}
                                              .set(cpf.O_out_body())
                                              .set_lhs(cpf.I_in_body())
                                              .set_rhs(cpf.F_in_body()))
                                     .finalize());
            }

            // ConvolutionBackpropFilters implements the derivative of a convolution with respect to its filter
            // input.
            template <>
            void Impl<op::ConvolutionBackpropFilters>::operator()()
            {
                this->check_inputs(2);
                this->check_outputs(1);

                LogConvolution(op_input(0),
                               op_input(1),
                               op().get_inputs()[0].get_shape().size() - 2,
                               op().get_window_movement_strides_backward(),
                               op().get_window_dilation_strides_backward(),
                               op().get_padding_below_backward(),
                               op().get_padding_above_backward(),
                               op().get_data_dilation_strides_backward(),
                               1,
                               0,
                               0,
                               1,
                               1,
                               0,
                               false);

                const auto& image = op_input(0);
                const auto& output = op_input(1);
                auto image_dims = op().get_inputs()[0].get_shape().size() - 2;
                const auto& padding_above = op().get_padding_above_forward();
                const auto& padding_below = op().get_padding_below_forward();
                const auto& strides = op().get_window_movement_strides_forward();
                const auto& filter_dilation = op().get_window_dilation_strides_forward();
                const auto& data_dilation = op().get_data_dilation_strides_forward();
                const auto& filters_shape = op().get_filters_shape();

                ConvPoolFormatter cpf(image_dims,
                                      padding_below,
                                      padding_above,
                                      strides,
                                      filter_dilation,
                                      data_dilation,
                                      ConvPoolFormatter::OpType::Conv,
                                      ConvPoolFormatter::DerivType::Filter,
                                      filters_shape);

                this->set_output(start_tile_function()
                                     .add(cpf.I_in_header(image))
                                     .add(cpf.O_in_header(output))
                                     .add(cpf.F_out_header())
                                     .add(builder::BinaryContraction{"+", "*"}
                                              .set(cpf.F_out_body())
                                              .set_lhs(cpf.O_in_body())
                                              .set_rhs(cpf.I_in_body()))
                                     .finalize());
            }

            // ConvolutionBackpropData implements the derivative of a convolution with respect to its data
            // input.
            template <>
            void Impl<op::ConvolutionBackpropData>::operator()()
            {
                this->check_inputs(2);
                this->check_outputs(1);

                LogConvolution(op_input(0),
                               op_input(1),
                               op().get_inputs()[1].get_shape().size() - 2,
                               op().get_window_movement_strides_backward(),
                               op().get_window_dilation_strides_backward(),
                               op().get_padding_below_backward(),
                               op().get_padding_above_backward(),
                               op().get_data_dilation_strides_backward(),
                               0,
                               1,
                               0,
                               1,
                               0,
                               1,
                               true);

                auto image_dims = op().get_inputs()[0].get_shape().size() - 2;
                const auto& filter = op_input(0);
                const auto& output = op_input(1);
                const auto& padding_above = op().get_padding_above_forward();
                const auto& padding_below = op().get_padding_below_forward();
                const auto& strides = op().get_window_movement_strides_forward();
                const auto& filter_dilation = op().get_window_dilation_strides_forward();
                const auto& data_dilation = op().get_data_dilation_strides_forward();
                const auto& data_batch_shape = op().get_data_batch_shape();

                ConvPoolFormatter cpf(image_dims,
                                      padding_below,
                                      padding_above,
                                      strides,
                                      filter_dilation,
                                      data_dilation,
                                      ConvPoolFormatter::OpType::Conv,
                                      ConvPoolFormatter::DerivType::Data,
                                      data_batch_shape);

                this->set_output(start_tile_function()
                                     .add(cpf.F_in_header(filter))
                                     .add(cpf.O_in_header(output))
                                     .add(cpf.I_out_header())
                                     .add(builder::BinaryContraction{"+", "*"}
                                              .set(cpf.I_out_body())
                                              .set_lhs(cpf.O_in_body())
                                              .set_rhs(cpf.F_in_body()))
                                     .finalize());
            }

            template <typename O>
            inline void ConvolutionImpl<O>::LogConvolution(vertexai::plaidml::variable image,
                                                           vertexai::plaidml::variable filter,
                                                           std::size_t image_dims,
                                                           const Strides& window_movement_strides,
                                                           const Strides& window_dilation_strides,
                                                           const CoordinateDiff& padding_below,
                                                           const CoordinateDiff& padding_above,
                                                           const Strides& data_dilation_strides,
                                                           std::size_t batch_axis_data,
                                                           std::size_t input_channel_axis_data,
                                                           std::size_t input_channel_axis_filters,
                                                           std::size_t output_channel_axis_filters,
                                                           std::size_t batch_axis_result,
                                                           std::size_t output_channel_axis_result,
                                                           bool rotate_filter)
            {
                this->check_inputs(2);
                this->check_outputs(1);

                NGRAPH_DEBUG << "image_dims: " << image_dims;
                NGRAPH_DEBUG << "first_dims: " << this->op().get_inputs()[0].get_shape();
                NGRAPH_DEBUG << "second_dims: " << this->op().get_inputs()[1].get_shape();
                NGRAPH_DEBUG << "output_dims: " << this->op().get_outputs()[0].get_shape();
                NGRAPH_DEBUG << "padding_below: " << padding_below;
                NGRAPH_DEBUG << "padding_above: " << padding_above;
                NGRAPH_DEBUG << "window_movement_strides: " << window_movement_strides;
                NGRAPH_DEBUG << "window_dilation_strides: " << window_dilation_strides;
                NGRAPH_DEBUG << "data_dilation_strides:" << data_dilation_strides;
                NGRAPH_DEBUG << "batch_axis_data: " << batch_axis_data;
                NGRAPH_DEBUG << "input_channel_axis_data: " << input_channel_axis_data;
                NGRAPH_DEBUG << "input_channel_axis_filters: " << input_channel_axis_filters;
                NGRAPH_DEBUG << "output_channel_axis_filters: " << output_channel_axis_filters;
                NGRAPH_DEBUG << "batch_axis_result: " << batch_axis_result;
                NGRAPH_DEBUG << "output_channel_axis_result: " << output_channel_axis_result;
                NGRAPH_DEBUG << "rotate_filter: " << rotate_filter;
            }

            namespace
            {
                Impl<op::Convolution>::Registration register_convolution;
                Impl<op::ConvolutionBackpropFilters>::Registration
                    register_convolution_backprop_filters;
                Impl<op::ConvolutionBackpropData>::Registration register_convolution_backprop_data;
            }
        }
    }
}
