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

#include <memory>

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise addition operation.
        ///
        class Add : public util::BinaryElementwiseArithmetic
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs an unitialized addition operation
            Add() = default;

            /// \brief Constructs an addition operation.
            ///
            /// \param arg0 Output that produces the first input tensor.<br>
            /// `[d0, ...]`
            /// \param arg1 Output that produces the second input tensor.<br>
            /// `[d0, ...]`
            /// \param autob Auto broadcast specification
            ///
            /// Output `[d0, ...]`
            ///
            Add(const Output<Node>& arg0,
                const Output<Node>& arg1,
                const AutoBroadcastSpec& autob = AutoBroadcastSpec());

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            virtual bool is_commutative() override { return true; }
        };
    }

    std::shared_ptr<ngraph::Node> operator+(const std::shared_ptr<ngraph::Node> arg0,
                                            const std::shared_ptr<ngraph::Node> arg1);
}
