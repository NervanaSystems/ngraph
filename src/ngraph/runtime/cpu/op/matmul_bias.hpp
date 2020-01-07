//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"

namespace ngraph
{
    namespace op
    {
        class MatmulBias : public Op
        {
        public:
            CPU_BACKEND_API
            static constexpr NodeTypeInfo type_info{"MatmulBias", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            CPU_BACKEND_API MatmulBias(const Output<Node>& W,
                                       const Output<Node>& x,
                                       const Output<Node>& b,
                                       Shape shape_w,
                                       Shape shape_x,
                                       bool transpose_w,
                                       bool transpose_x,
                                       AxisSet axes = AxisSet{});

            void validate_and_infer_types() override;

            bool get_is_a_transposed() const { return m_transpose_w; }
            bool get_is_b_transposed() const { return m_transpose_x; }
            Shape get_a_shape() const { return m_shape_w; }
            Shape get_b_shape() const { return m_shape_x; }
            const AxisSet& get_broadcast_axes() const { return m_broadcast_axes; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            Shape m_shape_w;
            Shape m_shape_x;
            bool m_transpose_w;
            bool m_transpose_x;
            AxisSet m_broadcast_axes;
        };
    }
}
