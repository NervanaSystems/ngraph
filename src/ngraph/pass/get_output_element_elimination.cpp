/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/pad.hpp"
#include "ngraph/ops/product.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "get_output_element_elimination.hpp"

using namespace ngraph;

bool ngraph::pass::GetOutputElementElimination::run_on_function(std::shared_ptr<ngraph::Function> f)
{
	bool optimized = false;
	for (auto n : f->get_ordered_ops())
	{
		for (auto& input : n->get_inputs())
		{

			if (auto goe = std::dynamic_pointer_cast<op::GetOutputElement>(input.get_output().get_node()))
			{
                std::cout << "goe = " << goe->get_name() << std::endl;
                std::cout << "n = " << n->get_name() << std::endl;
                std::cout << "multi = " << goe->get_inputs().at(0).get_output().get_node()->get_name() << std::endl;
                //GetOutputElement always has one input
                //whose get_output should be a multi-output node
				input.replace_output(goe->get_inputs().at(0).get_output());
				optimized = true;
			}
		}
	}
    return optimized;
}
