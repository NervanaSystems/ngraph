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

#include <memory>

#include <iostream>

#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        class Trace : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a trace operation.
            ///
            /// \param arg Node that produces the input tensor.
            Trace(const std::shared_ptr<Node> arg,
                  Coordinate lower,
                  Coordinate upper,
                  Strides strides);

            Trace(const std::shared_ptr<Node> arg)
                : Trace(arg,
                        Coordinate(arg->get_shape().size(), 0),
                        Coordinate{arg->get_shape()},
                        Strides(arg->get_shape().size(), 1))
            {
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            Coordinate get_lower_bounds() const { return m_lower; }
            Coordinate get_upper_bounds() const { return m_upper; }
            Coordinate get_strides() const { return m_strides; }
            std::string get_trace_str() const
            {
                return const_cast<op::Trace*>(this)
                    ->get_inputs()
                    .at(0)
                    .get_output()
                    .get_node()
                    ->get_name();
            }

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            Coordinate m_lower;
            Coordinate m_upper;
            Strides m_strides;
        };
    }
}
