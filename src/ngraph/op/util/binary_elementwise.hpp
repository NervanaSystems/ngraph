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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            class NGRAPH_API BinaryElementwise : public Op
            {
            protected:
                BinaryElementwise(const AutoBroadcastSpec& autob);

                /// \brief Constructs a binary elementwise arithmetic operation.
                ///
                /// \param arg0 Output that produces the first input tensor.
                /// \param arg1 Output that produces the second input tensor.
                BinaryElementwise(const Output<Node>& arg0,
                                            const Output<Node>& arg1,
                                            const AutoBroadcastSpec& autob);

                void validate_and_infer_elementwise_args();

                const AutoBroadcastSpec& get_autob() const override { return m_autob; }
                void set_autob(const AutoBroadcastSpec& autob) { m_autob = autob; }
                bool supports_auto_broadcast() const override { return true; }
                bool is_binary_elementwise_logical() const override { return true; }
                bool visit_attributes(AttributeVisitor& visitor) override;

            private:
                AutoBroadcastSpec m_autob;
            };
        }
    }
}
