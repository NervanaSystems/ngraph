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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace fluid
    {
        /// \brief Fluid pool
        class NGRAPH_API Pool : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"FluidPool", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            Pool() = default;
            /// \brief Constructs a Pool operation.
            ///
            /// \param x Input x
            Pool(const Output<Node>& x,
                 const Shape& window_shape,
                 const Strides& window_movement_strides,
                 const Shape& padding,
                 const bool global_pooling,
                 const bool ceil_mode,
                 const bool exclusive,
                 const bool adaptive,
                 const string pooling_type);

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const Shape& get_window_shape() const { return m_window_shape; }
            void set_window_shape(const Shape& window_shape) { m_window_shape = window_shape; }
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            void set_window_movement_strides(const Strides& window_movement_strides)
            {
                m_window_movement_strides = window_movement_strides;
            }
            const Shape& get_padding() const { return m_padding; }
            void set_padding(const Shape& padding) { m_padding = padding; }
            bool get_global_pooling() const { return m_global_pooling; }
            bool get_ceil_mode() const { return m_ceil_mode; }
            bool get_exclusive() const { return m_exclusive; }
            bool get_adaptive() const { return m_adaptive; }
            const string get_pooling_type() const { return m_pooling_type; }
        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding;
            bool m_global_pooling;
            bool m_ceil_mode;
            bool m_exclusive;
            bool m_adaptive;
            string m_pooling_type;
        };

        /// \brief Fluid reduce_sum_grad
        class NGRAPH_API PoolGrad : public ngraph::op::util::FusedOp
        {
        public:
            static constexpr NodeTypeInfo type_info{"FluidPoolGrad", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            PoolGrad() = default;

            /// \brief Constructs a PoolGrad operation.
            ///
            /// \param x Input tensor
            PoolGrad(const Output<Node>& x,
                     const Output<Node>& output,
                     const Output<Node>& output_delta,
                     const Shape& window_shape,
                     const Strides& window_movement_strides,
                     const Shape& padding,
                     const bool global_pooling,
                     const bool exclusive,
                     const bool adaptive,
                     const string pooling_type);

            virtual NodeVector decompose_op() const override;

            virtual void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const Shape& get_window_shape() const { return m_window_shape; }
            void set_window_shape(const Shape& window_shape) { m_window_shape = window_shape; }
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            void set_window_movement_strides(const Strides& window_movement_strides)
            {
                m_window_movement_strides = window_movement_strides;
            }
            const Shape& get_padding() const { return m_padding; }
            void set_padding(const Shape& padding) { m_padding = padding; }
            bool get_global_pooling() const { return m_global_pooling; }
            bool get_exclusive() const { return m_exclusive; }
            bool get_adaptive() const { return m_adaptive; }
            const string get_pooling_type() const { return m_pooling_type; }
        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding;
            bool m_global_pooling;
            bool m_exclusive;
            bool m_adaptive;
            string m_pooling_type;
        };
    }
}
