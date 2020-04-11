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

#include "constant_folding.hpp"

using namespace std;
using namespace ngraph;

bool ngraph::pass::revalidate_and_ensure_static(shared_ptr<Node> n)
{
    n->revalidate_and_infer_types();
    for (auto& o : n->outputs())
    {
        if (o.get_partial_shape().is_dynamic() || o.get_element_type().is_dynamic())
        {
            return false;
        }
    }
    return true;
}

void ngraph::pass::ConstantFolding::construct_constant_default()
{
    add_handler(
        "Constant folding defaults",
        [](const std::shared_ptr<Node>& node) -> bool {
            OutputVector replacements = node->constant_fold_default();
            if (replacements.empty())
            {
                return false;
            }
            NGRAPH_CHECK(replacements.size() == node->get_output_size(),
                         "constant_fold_default returned incorrect number of replacements for ",
                         node);
            bool result{false};
            for (size_t i = 0; i < replacements.size(); ++i)
            {
                auto node_output = node->output(i);
                auto replacement = replacements.at(i);
                if (node_output != replacement)
                {
                    cerr << "Replace " << node->output(i) << " with " << replacement << endl;
                    node->output(i).replace(replacement);
                    result = true;
                }
            }
            return result;
        },
        PassProperty::CHANGE_DYNAMIC_STATE);
}
