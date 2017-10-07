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

#include <memory>

namespace ngraph
{
    namespace pattern
    {
        class Matcher
        {
        public:
            Matcher() : m_is_valid(false), m_is_match(true) {}
            virtual ~Matcher() {}
            /// Called whern the pattern node matches a graph node.
            virtual void on_match_class(std::shared_ptr<Node>& pattern_node,
                                        const std::shared_ptr<Node>& graph_node,
                                        bool is_match) = 0;


            bool match(std::shared_ptr<Node>& pattern_node, const std::shared_ptr<Node>& graph_node);

        private:
            void match_arguments(const Nodes& pattern_args, const Nodes& args);
            bool m_is_valid;
            bool m_is_match;
        };
    }
}
