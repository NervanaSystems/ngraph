// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

namespace ngraph
{

    class ConvertOp : public BuiltinOp
    {
    public:
        ConvertOp(const Node::ptr& arg, const ngraph::element::Type& element_type)
            : BuiltinOp({arg})
            , m_element_type(element_type)
        {
        }

        virtual std::string op_name() const override { return "convert"; }
        virtual void        propagate_types() override;
    protected:
        const ngraph::element::Type& m_element_type;
    };


    namespace op
    {
        std::shared_ptr<ngraph::ConvertOp> convert(const Node::ptr& arg, const ngraph::element::Type& element_type);
    }
}
