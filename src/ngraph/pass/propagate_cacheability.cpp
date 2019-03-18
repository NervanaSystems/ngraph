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

#include "ngraph/pass/propagate_cacheability.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"

using namespace std;
using namespace ngraph;

bool pass::PropagateCacheability::run_on_function(shared_ptr<Function> function)
{
    for (auto& node : function->get_ordered_ops())
    {
        if (node->is_op())
        {
            auto op = static_pointer_cast<op::Op>(node);
            NGRAPH_DEBUG << "propagate cacheability: node is " << node->get_name();
            auto op_annotations = op->get_op_annotations();
            if (!op_annotations)
            {
                NGRAPH_DEBUG << "propagate cacheability: create op_annotations";
                op_annotations = op_annotations_factory();
                op->set_op_annotations(op_annotations);
            }
            if (node->is_parameter())
            {
                auto parameter = static_pointer_cast<op::Parameter>(node);
                op_annotations->set_cacheable(parameter->get_cacheable());
                NGRAPH_DEBUG << "propagate cacheability: cacheability is "
                             << parameter->get_cacheable();
            }
            else
            {
                bool cacheable = true;
                for (size_t i = 0; i < node->get_input_size(); i++)
                {
                    auto source_node = node->get_input_source_output(i).get_node();
                    NGRAPH_DEBUG << "propagate cacheability: source_node is " << source_node->get_name();
                    if (source_node->is_op())
                    {
                        auto source_node_op = static_pointer_cast<op::Op>(source_node);
                        auto source_node_op_annotations = source_node_op->get_op_annotations();
                        NGRAPH_ASSERT(source_node_op_annotations);
                        if (!source_node_op_annotations->is_cacheable())
                        {
                            cacheable = false;
                            break;
                        }
                    }
                }
                NGRAPH_DEBUG << "propagate cacheability: cacheability is " << cacheable;
                op_annotations->set_cacheable(cacheable);
            }
        }
    }
    return false;
}
