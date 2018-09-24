/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        class QuantizedConvolution : public Op
        {
        public:
            QuantizedConvolution(const std::shared_ptr<Node>& data_batch,
                                 const std::shared_ptr<Node>& filters,
                                 const Strides& window_movement_strides,
                                 const Strides& window_dilation_strides,
                                 const CoordinateDiff& padding_below,
                                 const CoordinateDiff& padding_above,
                                 const Strides& data_dilation_strides,
                                 const std::shared_ptr<Node> min_input,
                                 const std::shared_ptr<Node> max_input,
                                 const std::shared_ptr<Node> min_filter,
                                 const std::shared_ptr<Node> max_filter,
                                 const std::shared_ptr<Node> min_freezed_output,
                                 const std::shared_ptr<Node> max_freezed_output);
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            std::shared_ptr<Node> get_filters() { return get_argument(1); }
            std::shared_ptr<Node> get_data_batch() { return get_argument(0); }
            float get_input_min() const { return m_input_min; }
            float get_input_max() const { return m_input_max; }
            float get_filter_min() const { return m_filter_min; }
            float get_filter_max() const { return m_filter_max; }
            float get_freezed_output_min() const { return m_freezed_output_min; }
            float get_freezed_output_max() const { return m_freezed_output_max; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;
            float m_input_min;
            float m_input_max;
            float m_filter_min;
            float m_filter_max;
            float m_freezed_output_min;
            float m_freezed_output_max;
        };
    }
}
