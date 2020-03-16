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

#include <limits>
#include <list>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ngraph/pass/pass.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPUMemoryAssignment;
            }
        }
    }
}
class ngraph::runtime::cpu::pass::CPUMemoryAssignment : public ngraph::pass::FunctionPass
{
public:
    CPUMemoryAssignment(
        std::unordered_map<size_t, std::pair<TensorRole, std::unordered_set<descriptor::Tensor*>>>&,
        std::unordered_map<descriptor::Tensor*, size_t>&,
        size_t alignment = 1,
        bool disable_memory_sharing = false);
    bool run_on_function(std::shared_ptr<ngraph::Function>) override;

private:
    // Find in-place concat ops and set appropriate memory pool offset for its arguments
    void process_in_place_concat(std::vector<std::shared_ptr<Node>> nodes);

    // For a chain of concat ops, propagate memory pool offsets
    void propagate_in_place_concat(const ngraph::Output<ngraph::Node>& concat);

    // Find in-place slice ops and set appropriate memory pool offset for its output
    void process_in_place_slice(std::vector<std::shared_ptr<Node>> nodes);

    // propagate slice when its arg comes from function input
    void propagate_in_place_slice(const ngraph::Input<ngraph::Node>& input);

    // build buffer sets maps
    void build_buffer_sets_maps(std::vector<std::shared_ptr<Node>>& ops);

    // liveness analysis to build new and free list for each node
    void liveness_analysis(std::vector<std::shared_ptr<Node>>& ops);

    size_t get_bufferID(descriptor::Tensor* tensor);

    size_t m_alignment;
    bool m_disable_memory_sharing;
    std::set<descriptor::Tensor*> m_tensor_caching;
    std::unordered_map<size_t,
                       std::pair<ngraph::TensorRole, std::unordered_set<descriptor::Tensor*>>>&
        m_bufferID_to_tensorSets;
    std::unordered_map<descriptor::Tensor*, size_t>& m_tensor_to_bufferID;
};
