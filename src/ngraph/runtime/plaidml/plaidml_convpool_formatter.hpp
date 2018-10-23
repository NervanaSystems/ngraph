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

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/runtime/plaidml/plaidml_builder.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            class ConvPoolFormatter;
        }
    }
}

class ngraph::runtime::plaidml::ConvPoolFormatter
{
public:
    enum class OpType
    {
        Conv,
        MaxPool,
        AvgPool
    };
    enum class DerivType
    {
        None,
        Data,
        Filter
    };

    // TODO: Data dilation?
    // TODO: Types for the dimensional data?

    // Convolution-style constructor
    ConvPoolFormatter(std::size_t rank,
                      const ngraph::CoordinateDiff& pad_below,
                      const ngraph::CoordinateDiff& pad_above,
                      const ngraph::Strides& strides,
                      const ngraph::Strides& filter_dilation,
                      const ngraph::Strides& data_dilation,
                      ConvPoolFormatter::OpType op,
                      ConvPoolFormatter::DerivType deriv,
                      const ngraph::Shape& deriv_output_shape = Shape());

    // Pool-style constructor
    ConvPoolFormatter(std::size_t rank,
                      const ngraph::CoordinateDiff& pad_below,
                      const ngraph::CoordinateDiff& pad_above,
                      const ngraph::Strides& strides,
                      const ngraph::Shape& window_shape,
                      ConvPoolFormatter::OpType op,
                      ConvPoolFormatter::DerivType deriv);

    // Formatted tensors
    builder::Input F_in_header(vertexai::plaidml::variable var);
    builder::Input I_in_header(vertexai::plaidml::variable var);
    builder::Input O_in_header(vertexai::plaidml::variable var);
    builder::Output F_out_header();
    builder::Output I_out_header();
    builder::Output O_out_header();
    builder::ContractionOutput F_out_body();
    builder::ContractionOutput I_out_body();
    builder::ContractionOutput O_out_body();
    builder::ContractionInput F_in_body();
    builder::ContractionInput I_in_body();
    builder::ContractionInput O_in_body();

    // Special Operations
    builder::UnaryContraction Broadcast_Ones();
    builder::UnaryContraction Count();
    builder::UnaryContraction PoolContraction();
    builder::TernaryContraction PoolDerivContraction();

    // Index names / formulas
    std::string c();
    std::string ci();
    std::string co();
    std::string n();
    std::vector<std::string> xfs();
    std::vector<std::string> xis();
    std::vector<std::string> xos();

    // Dimension names / formulas
    std::string C();
    std::string CI();
    std::string CO();
    std::string N();
    std::vector<std::string> XFs();
    std::vector<std::string> XIs();
    std::vector<std::string> XOs();

    // Tensor names
    std::string F();
    std::string I();
    std::string O();

private:
    std::size_t m_rank;
    ngraph::CoordinateDiff m_pad_below;
    ngraph::CoordinateDiff m_pad_above;
    ngraph::Strides m_strides;
    ngraph::Strides m_filter_dilation;
    ngraph::Strides m_data_dilation;
    ngraph::Shape m_window_shape;
    OpType m_op = OpType::Conv;
    DerivType m_deriv = DerivType::None;
    ngraph::Shape m_filters_shape;
    ngraph::Shape m_data_batch_shape;
    std::vector<std::string> m_xfs;
    std::vector<std::string> m_xis;
    std::vector<std::string> m_xos;
    std::vector<std::string> m_XFs;
    std::vector<std::string> m_XIs;
    std::vector<std::string> m_XOs;
};
