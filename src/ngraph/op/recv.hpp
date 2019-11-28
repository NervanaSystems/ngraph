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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API Recv : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Recv", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs an unitialized recv operation.
                Recv() = default;
                /// \brief Constructs a Recv operation.
                ///
                /// \param arg The node for tensor to receive data
                /// \param src_id the source id which could be rank or node id.
                Recv(const Output<Node>& arg, int src_id);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                int get_src_id() const;

            private:
                int m_src_id;
            };
        }
        using v0::Recv;
    }
}
