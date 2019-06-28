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

#pragma once

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace plaidml
        {
            namespace op
            {
                class Convolution;
                class ConvolutionBackpropData;
                class ConvolutionBackpropFilters;
            }
        }
    }
}

class ngraph::runtime::plaidml::op::Convolution final : public ngraph::op::Op
{
public:
    Convolution(std::shared_ptr<ngraph::op::Convolution> src,
                const NodeVector& args,
                AxisVector data_axes,
                AxisVector filters_axes,
                AxisVector output_axes);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const final;

    const std::shared_ptr<ngraph::op::Convolution>& get_src() const { return m_src; }
    const AxisVector& get_data_axes() const { return m_data_axes; }
    const AxisVector& get_filters_axes() const { return m_filters_axes; }
    const AxisVector& get_output_axes() const { return m_output_axes; }
private:
    std::shared_ptr<ngraph::op::Convolution> m_src;
    AxisVector m_data_axes;
    AxisVector m_filters_axes;
    AxisVector m_output_axes;
};

class ngraph::runtime::plaidml::op::ConvolutionBackpropData final : public ngraph::op::Op
{
public:
    ConvolutionBackpropData(std::shared_ptr<ngraph::op::ConvolutionBackpropData> src,
                            const NodeVector& args,
                            AxisVector filters_axes,
                            AxisVector output_axes,
                            AxisVector data_axes);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const final;

    const std::shared_ptr<ngraph::op::ConvolutionBackpropData>& get_src() const { return m_src; }
    const AxisVector& get_filters_axes() const { return m_filters_axes; }
    const AxisVector& get_output_axes() const { return m_output_axes; }
    const AxisVector& get_data_axes() const { return m_data_axes; }
private:
    std::shared_ptr<ngraph::op::ConvolutionBackpropData> m_src;
    AxisVector m_filters_axes;
    AxisVector m_output_axes;
    AxisVector m_data_axes;
};

class ngraph::runtime::plaidml::op::ConvolutionBackpropFilters final : public ngraph::op::Op
{
public:
    ConvolutionBackpropFilters(std::shared_ptr<ngraph::op::ConvolutionBackpropFilters> src,
                               const NodeVector& args,
                               AxisVector data_axes,
                               AxisVector output_axes,
                               AxisVector filters_axes);

    void validate_and_infer_types() final;

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const final;

    const std::shared_ptr<ngraph::op::ConvolutionBackpropFilters>& get_src() const { return m_src; }
    const AxisVector& get_data_axes() const { return m_data_axes; }
    const AxisVector& get_output_axes() const { return m_output_axes; }
    const AxisVector& get_filters_axes() const { return m_filters_axes; }
private:
    std::shared_ptr<ngraph::op::ConvolutionBackpropFilters> m_src;
    AxisVector m_data_axes;
    AxisVector m_output_axes;
    AxisVector m_filters_axes;
};
