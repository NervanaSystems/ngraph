// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        class Reshape : public IndexBuiltin
        {
        public:
            ///
            /// @param arg The tensor view to be reshaped.
            /// @param input_order The order in which to iterate over input axes. (TODO: that needs more explanation)
            ///        This must be a permutation of the sequence (0,...,n-1) where n is the rank of the input tensor.
            /// @param output_shape The output shape. If the input shape is (a0,...,ak-1) then the output shape must
            ///        be of the form (b0,...,bj-1) where product(ai) == product(bi).
            ///
            Reshape(const std::shared_ptr<Node>& arg,
                    const AxisVector& input_order,
                    const Shape& output_shape)
                : IndexBuiltin(arg)
                , m_input_order(input_order)
                , m_output_shape(output_shape)
            {
            }

            virtual std::string description() const override { return "Reshape"; }
            virtual void propagate_types() override;

            const AxisVector& get_input_order() const { return m_input_order; }
            const Shape& get_output_shape() const { return m_output_shape; }
        protected:
            const AxisVector m_input_order;
            const Shape m_output_shape;
        };
    }
}
