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

#include <memory>

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise addition operation.
            ///
            class NGRAPH_API Add : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Add", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an uninitialized addition operation
                Add()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastType::DEFAULT)
                {
                }

                /// \brief Constructs an addition operation.
                ///
                /// \param arg0 Output that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Output that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                Add(const Output<Node>& arg0,
                    const Output<Node>& arg1,
                    const AutoBroadcastSpec& auto_broadcast = AutoBroadcastType::DEFAULT);

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual bool is_commutative() const override { return true; }
                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
            };
        }
    }

    NGRAPH_API
    std::shared_ptr<Node> operator+(const Output<Node>& arg0, const Output<Node>& arg1);
}
