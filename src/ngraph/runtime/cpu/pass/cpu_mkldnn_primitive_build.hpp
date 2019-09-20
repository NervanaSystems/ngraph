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

#include <fstream>
#include <functional>
#include <typeindex>
#include <unordered_map>

#define CONSTRUCT_PRIMITIVE_BUILD_STRING_DECL(op_name)                                             \
    construct_primitive_build_string<op_name>(ngraph::runtime::cpu::MKLDNNEmitter &                \
                                                  mkldnn_emitter,                                  \
                                              ngraph::Node * node,                                 \
                                              std::string & construct_string,                      \
                                              std::vector<size_t> & deps,                          \
                                              size_t & index,                                      \
                                              size_t & scratchpad_size,                            \
                                              std::ofstream & desc_file)

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
                using PrimitiveBuildStringConstructFunction =
                    std::function<void(ngraph::runtime::cpu::MKLDNNEmitter&,
                                       ngraph::Node*,
                                       std::string&,
                                       std::vector<size_t>&,
                                       size_t&,
                                       size_t&,
                                       std::ofstream&)>;
                using PrimitiveBuildStringConstructOpMap =
                    std::unordered_map<std::type_index, PrimitiveBuildStringConstructFunction>;

                /// This pass traverses the call graph and creates MKLDNN primitives for those ops
                /// that have been assigned to MKLDNN.
                class MKLDNNPrimitiveBuildPass : public ngraph::pass::CallGraphPass
                {
                private:
                    std::string m_desc_filename;

                    ngraph::runtime::cpu::MKLDNNEmitter& m_mkldnn_emitter;

                    /// External map to store each node with mkldnn implementation and its mkldnn
                    /// creation string, deps, mkldnn primitive index, and mkldnn primitive
                    /// scratchpad size.
                    std::map<const Node*,
                             std::tuple<std::string, std::vector<size_t>, size_t, size_t>>&
                        m_node_primitive_string_deps_index_size_map;

                public:
                    MKLDNNPrimitiveBuildPass(
                        std::string filename,
                        ngraph::runtime::cpu::MKLDNNEmitter& mkldnn_emitter,
                        std::map<const Node*,
                                 std::tuple<std::string, std::vector<size_t>, size_t, size_t>>&
                            node_primitive_string_deps_index_size_map)
                        : m_desc_filename(filename)
                        , m_mkldnn_emitter(mkldnn_emitter)
                        , m_node_primitive_string_deps_index_size_map(
                              node_primitive_string_deps_index_size_map)
                    {
                    }

                    bool run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes) override;

                    template <typename OP>
                    static void construct_primitive_build_string(
                        ngraph::runtime::cpu::MKLDNNEmitter& /* mkldnn_emitter */,
                        ngraph::Node* node,
                        std::string& /* construct_string */,
                        std::vector<size_t>& /* deps */,
                        size_t& /* index */,
                        size_t& /* scratchpad size */,
                        std::ofstream& /* desc_file */)
                    {
                        throw std::runtime_error("Unimplemented op '" + node->description() +
                                                 "' in MKLDNNPrimitiveBuildPass");
                    }
                };
            }
        }
    }
}
