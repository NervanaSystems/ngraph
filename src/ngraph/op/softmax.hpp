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

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Softmax operation.
        ///
        class Softmax : public util::UnaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs a softmax operation.
            ///
            /// \param arg Node that produces the first input tensor.<br>
            /// `[d0, ...]`
            /// \param axes The axis positions (0-based) on which to calculate the softmax.
            ///
            /// Output `[d0, ...]`
            ///
            Softmax(const std::shared_ptr<Node>& arg, const AxisSet& axes)
                : UnaryElementwiseArithmetic("Softmax", arg)
                , m_axes(axes)
            {
                for (auto axis : m_axes)
                {
                    if (axis >= get_shape().size())
                    {
                        throw ngraph_error("Axis for softmax reduction operator is out of bounds");
                    }
                }

                // empty axes == all axes
                if (m_axes.size() == 0)
                {
                    for (size_t i = 0; i < get_shape().size(); ++i)
                    {
                        m_axes.insert(i);
                    }
                }
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override
            {
                if (new_args.size() != 1)
                {
                    throw ngraph_error("Incorrect number of new arguments");
                }
                return std::make_shared<Softmax>(new_args.at(0), m_axes);
            }

            const AxisSet& get_axes() const { return m_axes; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;

        private:
            AxisSet m_axes;
        };
    }
}
