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

#include "ngraph/op/util/binary_elementwise_logical.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise logical-and operation.
        ///
        class And : public util::BinaryElementwiseLogical
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a logical-and operation.
            And() = default;

            /// \brief Constructs a logical-and operation.
            ///
            /// \param arg0 Output that produces the first input tensor.<br>
            /// `[d0, ...]`
            /// \param arg1 Output that produces the second input tensor.<br>
            /// `[d0, ...]`
            /// \param autob Auto broadcast specification
            ///
            /// Output `[d0, ...]`
            ///
            And(const Output<Node>& arg0,
                const Output<Node>& arg1,
                const AutoBroadcastSpec& autob = AutoBroadcastSpec());

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual bool is_commutative() override { return true; }
        };
    }
}
