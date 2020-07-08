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

#include <functional>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            class Label;
            class Pattern;
        }

        class Matcher;
        class MatchState;

        using RPatternValueMap = std::map<std::shared_ptr<Node>, OutputVector>;
        using PatternValueMap = std::map<std::shared_ptr<Node>, Output<Node>>;
        using PatternValueMaps = std::vector<PatternValueMap>;

        using PatternMap = std::map<std::shared_ptr<Node>, std::shared_ptr<Node>>;

        PatternMap as_pattern_map(const PatternValueMap& pattern_value_map);
        PatternValueMap as_pattern_value_map(const PatternMap& pattern_map);

        template <typename T>
        std::function<bool(Output<Node>)> has_class()
        {
            auto pred = [](Output<Node> node) -> bool {
                return is_type<T>(node.get_node_shared_ptr());
            };

            return pred;
        }

        namespace op
        {
            using ValuePredicate = std::function<bool(const Output<Node>& value)>;
        }
    }
}

class NGRAPH_API ngraph::pattern::op::Pattern : public Node
{
public:
    /// \brief \p a base class for all patterns
    ///
    Pattern(const OutputVector& patterns, ValuePredicate pred)
        : Node(patterns)
        , m_predicate(pred)
    {
        if (!m_predicate)
        {
            m_predicate = [](const Output<Node>&) { return true; };
        }
    }

    Pattern(const OutputVector& patterns)
        : Pattern(patterns, nullptr)
    {
    }

    ValuePredicate get_predicate() const;

    bool is_pattern() const override { return true; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override
    {
        throw std::runtime_error("Patterns do not support cloning");
    }

protected:
    ValuePredicate m_predicate;
};
