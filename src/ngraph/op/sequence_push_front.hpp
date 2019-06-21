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

namespace ngraph
{
    namespace op
    {
        class SequencePushFront : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Empty op
            SequencePushFront() = default;

            /// \brief Prepend a value to a sequence
            /// \param value The value to prepend
            /// \param sequence The sequence
            SequencePushFront(const Output<Node>& value,
                              const Output<Node>& sequence,
                              const AutoBroadcastSpec& autob = AutoBroadcastSpec());

            void validate_and_infer_types() override;
            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
            const AutoBroadcastSpec& get_autob() const { return m_autob; }
            void set_autob(const AutoBroadcastSpec& autob) { m_autob = autob; }
        private:
            AutoBroadcastSpec m_autob;
        };
    }
}
