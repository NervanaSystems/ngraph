//*****************************************************************************
// Copyright 2020 Intel Corporation
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
#include "variable.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v3
        {
            class NGRAPH_API ReadValue : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReadValue", 3};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ReadValue() = default;

                ReadValue(const Output<Node>& new_value, std::string variable_id);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                bool visit_attributes(AttributeVisitor& visitor) override;

                std::string get_variable_id() { return m_variable_id; }
                std::shared_ptr<::ngraph::v3::Variable> get_variable() { return m_variable; }
            private:
                std::string m_variable_id;
                std::shared_ptr<::ngraph::v3::Variable> m_variable;
            };
        }
        using v3::ReadValue;
    }
}
