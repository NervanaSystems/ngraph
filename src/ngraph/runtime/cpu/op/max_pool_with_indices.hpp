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

#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        //MaxPoolWithIndices produces two outputs.
        //The first output is equivalent to what MaxPool produces
        //The second one contains the indices of the maximum numbers
        //for each window in input (arg)
        //These indices are used by MKLDNN for a back propagation pass
        class MaxPoolWithIndices : public util::RequiresTensorViewArgs
        {
        public:
            MaxPoolWithIndices(const std::shared_ptr<Node>& arg,
                               const Shape& window_shape,
                               const Strides& window_movement_strides,
                               const Shape& padding_below,
                               const Shape& padding_above);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const Shape& get_window_shape() const { return m_window_shape; }
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Shape& get_padding_below() const { return m_padding_below; }
            const Shape& get_padding_above() const { return m_padding_above; }
            virtual std::shared_ptr<Node> get_default_value() const override
            {
                return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
            }

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
        };

        //MaxPoolWithIndicesBackprop takes MaxPoolWithIndices' outputs and
        //pass the indices directly to MKLDNN to avoid max indices recomputation
        class MaxPoolWithIndicesBackprop : public util::RequiresTensorViewArgs
        {
        public:
            MaxPoolWithIndicesBackprop(const std::shared_ptr<Node>& arg_forward,
                                       const std::shared_ptr<Node>& delta,
                                       const std::shared_ptr<Node>& indices,
                                       const Shape& window_shape,
                                       const Strides& window_movement_strides,
                                       const Shape& padding_below,
                                       const Shape& padding_above);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const Shape& get_window_shape() const { return m_window_shape; }
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Shape& get_padding_below() const { return m_padding_below; }
            const Shape& get_padding_above() const { return m_padding_above; }
        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
        };
    }
}
