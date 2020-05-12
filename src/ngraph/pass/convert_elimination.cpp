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

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/pass/convert_elimination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/opsets/opset3.hpp"

using namespace std;
using namespace ngraph;

//************************
// Eliminates 2 types of spines with converts:
//
// 1. [ op1--> convert2(type i64->i32)--> ReduceMin--> convert1(type i32->i64)--> op2 ] to
//                    [ op1--> ReduceMin--> op2 ]
// 2. [ op3--> convert1-->  NonZero--> op4 ] to
//                    [ op3--> NonZero--> op4 ]
//
//*************************
static bool eliminate_convert(const std::shared_ptr<Node>& node)
{
    auto convert = as_type_ptr<opset3::Convert>(node);
    auto destination_type = convert->get_destination_type();
    auto input_op = convert->input_value(0);

    // case 1
    if (is_type<opset3::ReduceMin>(input_op.get_node()))
    {
        auto convert_2 = input_op.get_node()->input_value(0).get_node();
        if (is_type<opset3::Convert>(convert_2) && convert_2->get_users().size() == 1 &&
            destination_type == convert_2->get_input_element_type(0))
        {
            if (destination_type == element::i64 && convert_2->get_element_type() == element::i32)
            {
                // replace ReplaceMin as convert's output
                if (replace_output_update_name(convert->output(0), input_op))
                {
                    input_op = convert_2->input_value(0);
                    // replace input to convert_2 as convert_2's output
                    return replace_output_update_name(convert_2->output(0), input_op);
                }
            }
        }
    }
    // case 2
    if (convert->get_users().size() == 1)
    {
        static const std::set<NodeTypeInfo> type_agnostic{TI(opset3::NonZero)};
        if (type_agnostic.count(convert->get_users()[0]->get_type_info()) == 1)
        {
            return replace_output_update_name(convert->output(0), input_op);
        }
    }
    return false;
}

bool pass::ConvertElimination::run_on_function(std::shared_ptr<Function> function)
{
    bool clobbered = false;

    for (const auto& node : function->get_ops())
    {
        if (node->get_type_info() == TI(opset3::Convert))
        {
            clobbered = eliminate_convert(node) || clobbered;
        }
    }

    return clobbered;
}
