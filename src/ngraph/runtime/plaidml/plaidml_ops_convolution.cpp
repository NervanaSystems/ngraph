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

#include "ngraph/runtime/plaidml/plaidml_ops_convolution.hpp"
#include "ngraph/except.hpp"
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
            NGRAPH_PLAIDML_OP_CLASS(ImplConvolution, OpImpl<plaidml::op::Convolution>);
            NGRAPH_PLAIDML_OP_CLASS(ImplConvolutionBackpropFilters,
                                    OpImpl<plaidml::op::ConvolutionBackpropFilters>);
            NGRAPH_PLAIDML_OP_CLASS(ImplConvolutionBackpropData,
                                    OpImpl<plaidml::op::ConvolutionBackpropData>);
        }
    }
}

constexpr ngraph::NodeTypeInfo ngraph::runtime::plaidml::op::Convolution::type_info;

ngraph::runtime::plaidml::op::Convolution::Convolution(std::shared_ptr<ngraph::op::Convolution> src,
                                                       const OutputVector& args,
                                                       AxisVector data_axes,
                                                       AxisVector filters_axes,
                                                       AxisVector output_axes)
    : Op{args}
    , m_src{std::move(src)}
    , m_data_axes{std::move(data_axes)}
    , m_filters_axes{std::move(filters_axes)}
    , m_output_axes{std::move(output_axes)}
{
    constructor_validate_and_infer_types();
}

void ngraph::runtime::plaidml::op::Convolution::validate_and_infer_types()
{
    auto src_shape = m_src->get_output_shape(0);
    Shape out_shape(src_shape.size());
    for (std::size_t idx = 0; idx < src_shape.size(); ++idx)
    {
        out_shape[idx] = src_shape.at(m_output_axes.at(idx));
    }
    set_output_type(0, m_src->get_element_type(), out_shape);
}

std::shared_ptr<ngraph::Node>
    ngraph::runtime::plaidml::op::Convolution::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error{"PlaidMLConvolution requires two inputs (data and filters)"};
    }
    return std::make_shared<Convolution>(
        m_src, as_output_vector(new_args), m_data_axes, m_filters_axes, m_output_axes);
}

constexpr ngraph::NodeTypeInfo ngraph::runtime::plaidml::op::ConvolutionBackpropData::type_info;

ngraph::runtime::plaidml::op::ConvolutionBackpropData::ConvolutionBackpropData(
    std::shared_ptr<ngraph::op::ConvolutionBackpropData> src,
    const OutputVector& args,
    AxisVector filters_axes,
    AxisVector output_axes,
    AxisVector data_axes)
    : Op{args}
    , m_src{std::move(src)}
    , m_filters_axes{std::move(filters_axes)}
    , m_output_axes{std::move(output_axes)}
    , m_data_axes{std::move(data_axes)}
{
    constructor_validate_and_infer_types();
}

void ngraph::runtime::plaidml::op::ConvolutionBackpropData::validate_and_infer_types()
{
    auto src_shape = m_src->get_output_shape(0);
    Shape out_shape(src_shape.size());
    for (std::size_t idx = 0; idx < src_shape.size(); ++idx)
    {
        out_shape[idx] = src_shape.at(m_output_axes.at(idx));
    }
    set_output_type(0, m_src->get_element_type(), out_shape);
}

std::shared_ptr<ngraph::Node>
    ngraph::runtime::plaidml::op::ConvolutionBackpropData::copy_with_new_args(
        const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error{"PlaidMLConvolutionBackpropData requires two inputs (data and output)"};
    }
    return std::make_shared<ConvolutionBackpropData>(
        m_src, as_output_vector(new_args), m_filters_axes, m_output_axes, m_data_axes);
}

constexpr ngraph::NodeTypeInfo ngraph::runtime::plaidml::op::ConvolutionBackpropFilters::type_info;

ngraph::runtime::plaidml::op::ConvolutionBackpropFilters::ConvolutionBackpropFilters(
    std::shared_ptr<ngraph::op::ConvolutionBackpropFilters> src,
    const OutputVector& args,
    AxisVector data_axes,
    AxisVector output_axes,
    AxisVector filters_axes)
    : Op{args}
    , m_src{std::move(src)}
    , m_data_axes{std::move(data_axes)}
    , m_output_axes{std::move(output_axes)}
    , m_filters_axes{std::move(filters_axes)}
{
    constructor_validate_and_infer_types();
}

void ngraph::runtime::plaidml::op::ConvolutionBackpropFilters::validate_and_infer_types()
{
    auto src_shape = m_src->get_output_shape(0);
    Shape out_shape(src_shape.size());
    for (std::size_t idx = 0; idx < src_shape.size(); ++idx)
    {
        out_shape[idx] = src_shape.at(m_output_axes.at(idx));
    }
    set_output_type(0, m_src->get_element_type(), out_shape);
}

std::shared_ptr<ngraph::Node>
    ngraph::runtime::plaidml::op::ConvolutionBackpropFilters::copy_with_new_args(
        const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error{
            "PlaidMLConvolutionBackpropFilters requires two inputs (filters and output)"};
    }
    return std::make_shared<ConvolutionBackpropFilters>(
        m_src, as_output_vector(new_args), m_data_axes, m_output_axes, m_filters_axes);
}

// Convolution implements a standard ML convolultion, with optional striding, padding, and dilation.
void ngraph::runtime::plaidml::ImplConvolution::Apply()
{
    this->check_inputs(2);
    this->check_outputs(1);

    const auto& image = op_input(0);
    const auto& filter = op_input(1);
    auto rank = op().get_input_shape(0).size() - 2;
    const auto& padding_above = op().get_src()->get_padding_above();
    const auto& padding_below = op().get_src()->get_padding_below();
    const auto& strides = op().get_src()->get_window_movement_strides();
    const auto& filter_dilation = op().get_src()->get_window_dilation_strides();
    const auto& data_dilation = op().get_src()->get_data_dilation_strides();

    ConvPoolFormatter cpf{rank,
                          padding_below,
                          padding_above,
                          strides,
                          filter_dilation,
                          data_dilation,
                          ConvPoolFormatter::OpType::Conv,
                          ConvPoolFormatter::DerivType::None};

    this->set_output(start_tile_function()
                         .add(cpf.I_in_header(image).transpose(op().get_data_axes()))
                         .add(cpf.F_in_header(filter).transpose(op().get_filters_axes()))
                         .add(cpf.O_out_header())
                         .add(builder::BinaryContraction{"+", "*"}
                                  .set(cpf.O_out_body().transpose(op().get_output_axes()))
                                  .set_lhs(cpf.I_in_body().transpose(op().get_data_axes()))
                                  .set_rhs(cpf.F_in_body().transpose(op().get_filters_axes())))
                         .finalize());
}

// ConvolutionBackpropData implements the derivative of a convolution with respect to its data
// input.
void ngraph::runtime::plaidml::ImplConvolutionBackpropData::Apply()
{
    this->check_inputs(2);
    this->check_outputs(1);

    const auto& filter = op_input(0);
    const auto& output = op_input(1);
    auto rank = op().get_input_shape(0).size() - 2;
    const auto& padding_above = op().get_src()->get_padding_above_forward();
    const auto& padding_below = op().get_src()->get_padding_below_forward();
    const auto& strides = op().get_src()->get_window_movement_strides_forward();
    const auto& filter_dilation = op().get_src()->get_window_dilation_strides_forward();
    const auto& data_dilation = op().get_src()->get_data_dilation_strides_forward();
    const auto& data_batch_shape = op().get_src()->get_data_batch_shape();

    ConvPoolFormatter cpf{rank,
                          padding_below,
                          padding_above,
                          strides,
                          filter_dilation,
                          data_dilation,
                          ConvPoolFormatter::OpType::Conv,
                          ConvPoolFormatter::DerivType::Data,
                          data_batch_shape};

    this->set_output(start_tile_function()
                         .add(cpf.F_in_header(filter).transpose(op().get_filters_axes()))
                         .add(cpf.O_in_header(output).transpose(op().get_output_axes()))
                         .add(cpf.I_out_header())
                         .add(builder::BinaryContraction{"+", "*"}
                                  .set(cpf.I_out_body().transpose(op().get_data_axes()))
                                  .set_lhs(cpf.O_in_body().transpose(op().get_output_axes()))
                                  .set_rhs(cpf.F_in_body().transpose(op().get_filters_axes())))
                         .finalize());
}

// ConvolutionBackpropFilters implements the derivative of a convolution with respect to its filter
// input.
void ngraph::runtime::plaidml::ImplConvolutionBackpropFilters::Apply()
{
    this->check_inputs(2);
    this->check_outputs(1);

    const auto& image = op_input(0);
    const auto& output = op_input(1);
    auto rank = op().get_input_shape(0).size() - 2;
    const auto& padding_above = op().get_src()->get_padding_above_forward();
    const auto& padding_below = op().get_src()->get_padding_below_forward();
    const auto& strides = op().get_src()->get_window_movement_strides_forward();
    const auto& filter_dilation = op().get_src()->get_window_dilation_strides_forward();
    const auto& data_dilation = op().get_src()->get_data_dilation_strides_forward();
    const auto& filters_shape = op().get_src()->get_filters_shape();

    ConvPoolFormatter cpf{rank,
                          padding_below,
                          padding_above,
                          strides,
                          filter_dilation,
                          data_dilation,
                          ConvPoolFormatter::OpType::Conv,
                          ConvPoolFormatter::DerivType::Filter,
                          filters_shape};

    this->set_output(start_tile_function()
                         .add(cpf.I_in_header(image).transpose(op().get_data_axes()))
                         .add(cpf.O_in_header(output).transpose(op().get_output_axes()))
                         .add(cpf.F_out_header())
                         .add(builder::BinaryContraction{"+", "*"}
                                  .set(cpf.F_out_body().transpose(op().get_filters_axes()))
                                  .set_lhs(cpf.O_in_body().transpose(op().get_output_axes()))
                                  .set_rhs(cpf.I_in_body().transpose(op().get_data_axes())))
                         .finalize());
}
