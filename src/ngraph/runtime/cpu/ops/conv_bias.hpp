/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        class ConvolutionBias : public util::RequiresTensorViewArgs
        {
        public:
            ConvolutionBias(const std::shared_ptr<op::Convolution>& conv,
                            const std::shared_ptr<Node>& bias);

            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            std::shared_ptr<Node> get_bias() { return get_input_op(2); }
            std::shared_ptr<Node> get_filters() { return get_input_op(1); }
            std::shared_ptr<Node> get_data_batch() { return get_input_op(0); }
            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override;

            void generate_adjoints(autodiff::Adjoints& adjoints,
                                   const std::shared_ptr<Node>& delta) override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;

        private:
            ConvolutionBias(const std::shared_ptr<Node>& data_batch,
                            const std::shared_ptr<Node>& filters,
                            const std::shared_ptr<Node>& bias,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides,
                            const CoordinateDiff& padding_below,
                            const CoordinateDiff& padding_above,
                            const Strides& data_dilation_strides);
        };
    }
}
