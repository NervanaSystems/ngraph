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

#include <string>

#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    /// \brief Visits the attributes of a node.
    ///
    /// Attributes are the values set when building a graph which are not
    /// computed as the graph executes. Values computed from the graph topology and attributes
    /// during compilation are not attributes.
    class AttributeVisitor
    {
    public:
        virtual ~AttributeVisitor() {}
        virtual void on_attribute(const std::string& name, std::string& value) = 0;
        virtual void on_attribute(const std::string& name, element::Type& value) = 0;
        virtual void on_attribute(const std::string& name, PartialShape& value) = 0;
        virtual void on_attribute(const std::string& name, bool& value) = 0;
        virtual void on_attribute(const std::string& name, int64_t& value) = 0;
        virtual void on_attribute(const std::string& name, uint64_t& value) = 0;
    };
}