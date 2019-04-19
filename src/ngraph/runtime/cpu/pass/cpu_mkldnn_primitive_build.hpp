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

#include "ngraph/pass/pass.hpp"

#include <functional>
#include <typeindex>
#include <unordered_map>

#define BUILD_PRIMITIVE_DECL(op_name)                                                              \
    build_primitive<op_name>(ngraph::runtime::cpu::MKLDNNEmitter & mkldnn_emitter,                 \
                             ngraph::Node * node)

namespace mkldnn
{
    class primitive;
}

namespace ngraph
{
    class Node;

    namespace runtime
    {
        namespace cpu
        {
            class MKLDNNEmitter;

            namespace pass
            {
                using PrimitiveBuildFunction =
                    std::function<size_t(ngraph::runtime::cpu::MKLDNNEmitter&, ngraph::Node*)>;
                using PrimitiveBuildOpMap =
                    std::unordered_map<std::type_index, PrimitiveBuildFunction>;

                /// This pass traverses the call graph and creates MKLDNN primitives for those ops
                /// that have been assigned to MKLDNN.
                class MKLDNNPrimitiveBuildPass : public ngraph::pass::CallGraphPass
                {
                private:
                    ngraph::runtime::cpu::MKLDNNEmitter& m_mkldnn_emitter;

                    /// External map to store each node with mkldnn implementation and its mkldnn
                    /// associated primitive index.
                    std::unordered_map<const Node*, size_t>& m_node_primitive_idx_map;

                public:
                    MKLDNNPrimitiveBuildPass(
                        ngraph::runtime::cpu::MKLDNNEmitter& mkldnn_emitter,
                        std::unordered_map<const Node*, size_t>& node_primitive_idx_map)
                        : m_mkldnn_emitter(mkldnn_emitter)
                        , m_node_primitive_idx_map(node_primitive_idx_map)
                    {
                    }

                    bool run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes) override;

                    template <typename OP>
                    static size_t
                        build_primitive(ngraph::runtime::cpu::MKLDNNEmitter& mkldnn_emitter,
                                        ngraph::Node* node)
                    {
                        throw std::runtime_error("Unimplemented op '" + node->description() +
                                                 "' in MKLDNNPrimitiveBuildPass");
                    }
                };
            }
        }
    }
}
