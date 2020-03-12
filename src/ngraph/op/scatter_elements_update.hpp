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

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API ScatterElementsUpdate : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScatterElementsUpdate", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ScatterElementsUpdate() = default;
                /// \brief Constructs a ScatterElementsUpdate node

                /// \param data            Input data
                /// \param indices         Data entry index that will be updated
                /// \param updates         Update values
                /// \param axis            Axis to scatter on
                ScatterElementsUpdate(const Output<Node>& data,
                                      const Output<Node>& indices,
                                      const Output<Node>& updates,
                                      const Output<Node>& axis);

                virtual void validate_and_infer_types() override;
                virtual bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }
        using v0::ScatterElementsUpdate;
    }
}
