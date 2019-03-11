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

#include <set>

#include "get_output_element_elimination.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sum.hpp"

using namespace ngraph;
using namespace std;

bool pass::GetOutputElementElimination::run_on_node(shared_ptr<Node> n)
{
    bool optimized = false;

    for (size_t i = 0; i < n->get_input_size(); i++)
    {
        if (auto goe = dynamic_pointer_cast<op::GetOutputElement>(n->get_input_source_output(i).get_node()))
        {
            n->replace_input_source(i,goe->get_input_source_output(goe->get_n()));
            optimized = true;
        }
    }

    return optimized;
}
