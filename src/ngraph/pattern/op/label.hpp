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

#include "ngraph/node.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/pattern.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            class Label : public Pattern
            {
                using Pattern::Pattern; // inherit c-tors
            public:
                bool is_bound() { return (bool)m_bound; };
                std::shared_ptr<Node> get_bound_node() { return m_bound; }
                virtual std::string description() const override { return "Label"; }
                void reset() { m_bound.reset(); }
                void bind(std::shared_ptr<Node> n) { m_bound = n; }
            private:
                std::shared_ptr<Node> m_bound;
            };
        }
    }
}
