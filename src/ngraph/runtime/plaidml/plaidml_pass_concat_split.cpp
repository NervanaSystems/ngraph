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

#include "ngraph/runtime/plaidml/plaidml_pass_concat_split.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"

static std::size_t kMaxConcatInputs = 8;

ngraph::runtime::plaidml::pass::ConcatSplit::ConcatSplit()
{
    auto concat_op =
        std::make_shared<pattern::op::Label>(element::i8, Shape{}, [](std::shared_ptr<Node> node) {
            auto op = dynamic_cast<ngraph::op::Concat*>(node.get());
            return op != nullptr && kMaxConcatInputs < op->get_input_size();
        });

    auto callback = [](pattern::Matcher& m) {
        auto concat = std::static_pointer_cast<ngraph::op::Concat>(m.get_match_root());
        auto args = concat->get_arguments();

        while (1 < args.size())
        {
            NodeVector new_args;
            auto b = args.begin();
            auto e = args.end();
            while (b != e)
            {
                NodeVector::iterator p;
                if (e < b + kMaxConcatInputs)
                {
                    p = e;
                }
                else
                {
                    p = b + kMaxConcatInputs;
                }
                if (p - b == 1)
                {
                    new_args.emplace_back(*b);
                }
                else
                {
                    NodeVector sub_args;
                    for (auto n = b; n != p; ++n)
                    {
                        sub_args.push_back(*n);
                    }
                    new_args.emplace_back(std::make_shared<ngraph::op::Concat>(
                        std::move(sub_args), concat->get_concatenation_axis()));
                }
                b = p;
            }
            args = std::move(new_args);
        }

        replace_node(std::move(concat), args[0]);
        return true;
    };

    add_matcher(std::make_shared<pattern::Matcher>(concat_op), callback);
}
