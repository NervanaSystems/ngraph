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

#include "ngraph/runtime/plaidml/plaidml_pass_concat_elision.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/plaidml/plaidml_ops_replicate.hpp"

ngraph::runtime::plaidml::pass::ConcatElision::ConcatElision()
{
    auto concat_op =
        std::make_shared<pattern::op::Label>(element::i8, Shape{}, [](std::shared_ptr<Node> node) {
            return dynamic_cast<ngraph::op::Concat*>(node.get()) != nullptr;
        });

    auto callback = [](pattern::Matcher& m) {
        auto concat = std::static_pointer_cast<ngraph::op::Concat>(m.get_match_root());
        auto args = concat->get_arguments();

        // Elide one-argument concats.
        if (args.size() == 1)
        {
            replace_node(concat, args.at(0));
            return true;
        }

        // Check for a run of inputs from the same source -- if we see
        // one, we can simplify it, and otherwise, we already have the
        // best Concat we can make.
        {
            bool found_input_run = false;
            std::size_t prev_instance_id = concat->get_instance_id(); // This will never be an arg
            for (const auto& arg : args)
            {
                auto current_instance_id = arg->get_instance_id();
                if (current_instance_id == prev_instance_id)
                {
                    found_input_run = true;
                    break;
                }
                prev_instance_id = current_instance_id;
            }
            if (!found_input_run)
            {
                return false;
            }
        }

        // Merge runs with the same input into Replicate calls.
        NodeVector new_args;
        auto run_begin = args.begin();

        // N.B. There's at least one argument to concat at this point
        // (actually, two, but we only care that there's at least
        // one), so run_end is still valid after this incremenent.
        auto run_end = run_begin + 1;

        for (;;)
        {
            // Invariants:
            // * [run_begin..run_end) is a range of identical arguments
            // * run_begin < run_end (there's at least one member of the range).
            if (run_end == args.end() || *run_begin != *run_end)
            {
                // End of the range.
                if (run_end - run_begin == 1)
                {
                    new_args.emplace_back(*run_begin);
                }
                else
                {
                    new_args.emplace_back(std::make_shared<plaidml::op::Replicate>(
                        *run_begin, concat->get_concatenation_axis(), run_end - run_begin));
                }
                if (run_end == args.end())
                {
                    break;
                }
                run_begin = run_end;
            }
            ++run_end;
        }

        // Re-check for single-input concat.
        if (new_args.size() == 1)
        {
            replace_node(concat, new_args.at(0));
            return true;
        }

        // Build a replacement concat.
        auto new_concat =
            std::make_shared<ngraph::op::Concat>(new_args, concat->get_concatenation_axis());
        replace_node(std::move(concat), std::move(new_concat));
        return true;
    };

    add_matcher(std::make_shared<pattern::Matcher>(concat_op), callback);
}
