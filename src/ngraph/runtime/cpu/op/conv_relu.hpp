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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"

namespace ngraph
{
    namespace op
    {
        /// \brief Relu(Convolution) forward prop for batched convolution operation.
        class ConvolutionRelu : public Op
        {
        public:
            CPU_BACKEND_API
            static constexpr NodeTypeInfo type_info{"ConvolutionRelu", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            CPU_BACKEND_API ConvolutionRelu(const std::shared_ptr<op::Convolution>& conv);

            CPU_BACKEND_API ConvolutionRelu(const Output<Node>& data_batch,
                                            const Output<Node>& filters,
                                            const Strides& window_movement_strides,
                                            const Strides& window_dilation_strides,
                                            const CoordinateDiff& padding_below,
                                            const CoordinateDiff& padding_above,
                                            const Strides& data_dilation_strides);

            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            Output<Node> get_filters() { return input(1).get_source_output(); }
            Output<Node> get_data_batch() { return input(0).get_source_output(); }
            bool with_relu() const { return true; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;
        };
    }
}
