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
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Group Convolution
        class GroupConvolution : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            GroupConvolution() = default;
            GroupConvolution(const Output<Node>& data_batch,
                             const Output<Node>& filters,
                             const Strides& window_movement_strides,
                             const Strides& window_dilation_strides,
                             const CoordinateDiff& padding_below,
                             const CoordinateDiff& padding_above,
                             const Strides& data_dilation_strides,
                             const size_t groups,
                             const PadType& pad_type = PadType::EXPLICIT);

            Shape get_weights_dimensions() const;
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            Output<Node> get_filters() { return input_value(1); }
            Output<Node> get_data_batch() { return input_value(0); }
            size_t get_groups() const { return m_groups; }
            const PadType& get_pad_type() const { return m_pad_type; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;
            virtual void post_validate_and_infer_types() override;

            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;
            size_t m_groups{0};
            PadType m_pad_type{PadType::NOTSET};
        };
    }
}
