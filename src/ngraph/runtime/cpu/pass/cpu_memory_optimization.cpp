//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

/// In-place-concat optimization makes the argument nodes of a concatenation node use the concatenation node's memory buffer
/// for their outputs. As a result, we eliminate memory copies from the memory buffers of the argument nodes to
/// that of the concatenation node. When there is a chain of in place concatenation nodes, we propagate the
/// memory buffer starting from the last concatenation node. Not all concatenation nodes can be optimized. This pass
/// marks all the nodes that can be optimized.
///
/// Example1:
/// parameter1 parameter2        parameter3 parameter4        parameter5 parameter6
///    \          /                 \          /                 \          /
///         add1                        add2                         add3
///           \                           |                            /
///                                    concat
///
/// Before optimization: the result of add1 is stored to the memory buffer assigned to add1, same for add2 and add3;
///                      then those results are copied to the memory buffer assigned to concat.
/// After optimization: the result of add1 is stored to the memory buffer assigned to concat, same for add2 and add3.
///                     there is no need to copy those results.
///
///
/// Example2:
/// parameter1 parameter2      parameter3 parameter4
///    \          /               \          /
///        add1                       add2
///          \                         /
///                     concat1                     parameter5
///                      |     \                        /
///                      |                 add3
///                       \                 /
///                               concat
///
/// After optimization: the result of add1 is stored to the memory buffer assigned to concat, same for add2 and add3.

#include "ngraph/runtime/cpu/pass/cpu_memory_optimization.hpp"

#include "ngraph/descriptor/output.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

using namespace ngraph;

bool runtime::cpu::pass::CPUMemoryOptimization::run_on_function(std::shared_ptr<Function> function)
{
    for (auto n : function->get_ordered_ops())
    {
        if (auto concat = std::dynamic_pointer_cast<op::Concat>(n))
        {
            auto shape = concat->get_input_shape(0);
            auto axis = concat->get_concatenation_axis();
            auto product = 1;
            for (int i = 0; i < axis; i++)
            {
                product *= shape[i];
            }
            if (product != 1)
            {
                NGRAPH_DEBUG << "cpu_memory_optimization: The product of Concat's shape "
                                "before concat axis is not 1, no in place concat";
                continue;
            }

            bool in_place_concat = true;
            auto output_md = mkldnn_utils::get_output_mkldnn_md(n.get(), 0);
            auto output_format = static_cast<mkldnn::memory::format>(output_md.data.format);
            for (size_t i = 0; i < n->get_input_size(); i++)
            {
                auto input_md = mkldnn_utils::get_input_mkldnn_md(n.get(), i);
                auto input_format = static_cast<mkldnn::memory::format>(input_md.data.format);
                if (output_format != input_format)
                {
                    NGRAPH_DEBUG << "cpu_memory_optimization: input format is different from "
                                    "output format, no in place concat";
                    in_place_concat = false;
                    break;
                }
            }
            if (!in_place_concat)
            {
                continue;
            }

            AxisVector axis_list = ngraph::get_default_order(shape);

            auto index = 0;
            for (descriptor::Input& input : concat->get_inputs())
            {
                // no tensors with zero-sized dimensions after zero_dim_tensor_elimination
                NGRAPH_ASSERT(shape_size(input.get_shape()) != 0);

                // check if input layout is padded
                auto input_md = mkldnn_utils::get_input_mkldnn_md(n.get(), index);
                index++;
                if (mkldnn_utils::is_mkldnn_padded_layout(input_md, axis_list))
                {
                    NGRAPH_DEBUG
                        << "cpu_memory_optimization: padded input layout, no in place concat";
                    in_place_concat = false;
                    break;
                }

                const auto& output = input.get_output();
                auto arg = output.get_node();
                if (std::dynamic_pointer_cast<op::Constant>(arg) ||
                    std::dynamic_pointer_cast<op::Parameter>(arg))
                {
                    NGRAPH_DEBUG << "cpu_memory_optimization: " << arg->get_name()
                                 << ": constant or parameter, no in place concat";
                    in_place_concat = false;
                    break;
                }

                NGRAPH_ASSERT(arg->get_output_size() == 1);

                if (!std::dynamic_pointer_cast<op::Concat>(arg))
                {
                    if (auto op = std::dynamic_pointer_cast<op::Op>(arg))
                    {
                        auto annotation = op->get_op_annotations();
                        if (annotation && annotation->get_in_place_oi_pairs().size() > 0)

                        {
                            NGRAPH_DEBUG << "cpu_memory_optimization: " << arg->get_name()
                                         << ": in place non-concat op, no in place concat";
                            in_place_concat = false;
                            break;
                        }
                    }
                }

                if (output.get_inputs().size() != 1)
                {
                    // check if we can do in place concat
                    auto concat_count = 0;
                    for (auto output_input : output.get_inputs())
                    {
                        auto user = output_input->get_node();
                        if (std::dynamic_pointer_cast<op::Concat>(user))
                        {
                            concat_count++;
                            if (concat_count == 2)
                            {
                                NGRAPH_DEBUG << "cpu_memory_optimization: multiple "
                                                "concat users, no in place concat";
                                in_place_concat = false;
                                break;
                            }
                        }
                    }
                    if (!in_place_concat)
                    {
                        break;
                    }

                    for (auto user : arg->get_users())
                    {
                        if ((user != concat))
                        {
                            if (auto op = std::dynamic_pointer_cast<op::Op>(user))
                            {
                                if (auto op_annotations = op->get_op_annotations())
                                {
                                    if (op_annotations->get_in_place_oi_pairs().size() > 0)
                                    {
                                        NGRAPH_DEBUG << "cpu_memory_optimization: "
                                                        "in place oi, no in place concat";
                                        in_place_concat = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if (!in_place_concat)
                    {
                        break;
                    }
                    else if (!is_post_dominated(arg.get(), n.get()))
                    {
                        NGRAPH_DEBUG << "cpu_memory_optimization: "
                                        "not post dominated, no in place concat";
                        in_place_concat = false;
                        break;
                    }
                }
            }

            if (in_place_concat)
            {
                auto op_annotations = concat->get_op_annotations();
                if (op_annotations)
                {
                    op_annotations->add_in_place_oi_pair({0, 0, false});
                }
                else
                {
                    op_annotations = std::make_shared<ngraph::runtime::cpu::CPUOpAnnotations>();
                    op_annotations->add_in_place_oi_pair({0, 0, false});
                    concat->set_op_annotations(op_annotations);
                }
            }
        }
    }
    return false;
}
