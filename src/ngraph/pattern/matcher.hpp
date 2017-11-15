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

#include <memory.h>
#include "ngraph/node.hpp"

namespace ngraph
{
    namespace pass
    {
        class GraphRewrite;
    }

    namespace pattern
    {
        using gr_callback_fn = std::function<void(class Matcher& m)>;

        namespace op
        {
            class Label;
        }

        class Matcher
        {
        public:
            Matcher(const std::shared_ptr<Node> pattern_node = nullptr,
                    gr_callback_fn callback = nullptr)
                : m_match_root(nullptr)
                , m_pattern_node(pattern_node)
                , m_callback(callback)
                , m_depth(0)
            {
            }
            virtual ~Matcher() {}
            // Called when the pattern node matches a graph node.
            virtual void on_match_class(const std::shared_ptr<Node>& pattern_node,
                                        const std::shared_ptr<Node>& graph_node,
                                        bool is_match);

            bool match(const std::shared_ptr<Node>& graph_node)
            {
                return match(m_pattern_node, graph_node);
            }

            bool match(const std::shared_ptr<Node>& pattern_node, //keep public for testing for now
                       const std::shared_ptr<Node>& graph_node);

            void process_match(gr_callback_fn callback = nullptr);

            static std::string pad(size_t num) { return std::string(num, ' '); }
            void reset() {}
            bool is_match() { return m_match_root != nullptr; }
            std::shared_ptr<Node> pattern_node() { return m_pattern_node; }
            std::shared_ptr<Node> match_root()
            {
                assert(is_match());
                return m_match_root;
            }

            void reset_pattern_nodes(std::shared_ptr<Node> node);

            friend op::Label; //TODO: refine to match_class

        protected:
            void virtual match_class(const std::shared_ptr<Node>& pattern_node,
                                     const std::shared_ptr<Node>& graph_node);

        private:
            void match_arguments(const Nodes& pattern_args, const Nodes& args);
            void match_pattern(const std::shared_ptr<Node>& pattern_node,
                               const std::shared_ptr<Node>& graph_node);
            void match_any(const std::shared_ptr<Node>& pattern_node,
                           const std::shared_ptr<Node>& graph_node);
            std::shared_ptr<Node> m_match_root;
            std::shared_ptr<Node> m_pattern_node;
            gr_callback_fn m_callback;
            size_t m_depth;
        };
    }
}
