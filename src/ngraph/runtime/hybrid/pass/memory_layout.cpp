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

#include <exception>
#include <sstream>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/hybrid/pass/memory_layout.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::hybrid::pass::MemoryLayout::MemoryLayout(size_t alignment)
    : m_alignment(alignment)
{
    if (m_alignment == 0)
    {
        throw invalid_argument("Memory alignment must be > 0");
    }
}

bool runtime::hybrid::pass::MemoryLayout::run_on_function(shared_ptr<ngraph::Function> function)
{
    ngraph::pass::MemoryManager mm(m_alignment, false);
    for (shared_ptr<Node> node : function->get_ordered_ops())
    {
        std::map<descriptor::Tensor*, descriptor::Tensor*> in_place_outputs;
        std::set<const descriptor::Tensor*> reused_inputs;

        if (node->is_op())
        {
            auto op = std::static_pointer_cast<op::Op>(node);
            // concat and slice in_place_oi should be treated differently
            if (!std::dynamic_pointer_cast<op::Concat>(node) &&
                !std::dynamic_pointer_cast<op::Slice>(node))
            {
                if (auto op_annotations = op->get_op_annotations())
                {
                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                    {
                        auto output = &node->get_outputs().at(oi_pair.output).get_tensor();
                        auto input = &node->get_inputs().at(oi_pair.input).get_tensor();
                        auto input_node =
                            node->get_inputs().at(oi_pair.input).get_output().get_node();

                        // For destructive kernel, this should be the last use
                        // Non-destructive kernels can pass through if memory sharing is disabled
                        if ((node->liveness_free_list.count(input) != 0 ||
                             std::dynamic_pointer_cast<op::GetOutputElement>(node)) &&
                            node->liveness_new_list.count(output) != 0)
                        {
                            in_place_outputs.insert({output, input});
                            reused_inputs.insert(input);
                        }
                    }
                }
            }
        }

        for (descriptor::Tensor* tensor : node->liveness_new_list)
        {
            size_t offset = in_place_outputs.count(tensor)
                                ? in_place_outputs.at(tensor)->get_pool_offset()
                                : mm.allocate(tensor->size());
            tensor->set_pool_offset(offset);
        }

        for (const descriptor::Tensor* tensor : node->liveness_free_list)
        {
            if (reused_inputs.count(tensor) == 0)
            {
                mm.free(tensor->get_pool_offset());
            }
        }
    }
    function->set_temporary_pool_size(mm.max_allocated());

    return false;
}
