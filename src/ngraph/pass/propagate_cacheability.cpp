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

#include "ngraph/pass/propagate_cacheability.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/runtime/cpu/cpu_op_annotations.hpp"


using namespace ngraph;

bool ngraph::pass::PropagateCacheability::run_on_function(std::shared_ptr<Function> function)
{
    for (auto& node : function->get_ordered_ops())
    {
    	if (auto op = std::dynamic_pointer_cast<op::Op>(node))
    	{
    		std::cout << "node is " << node->get_name() << std::endl;
    		auto op_annotations = op->get_op_annotations();
    		if (!op_annotations)
    		{
    			std::cout << "create op_annotations\n";
    			op_annotations = std::make_shared<ngraph::op::util::OpAnnotations>();
    			op->set_op_annotations(op_annotations);
    		}
    		if (std::dynamic_pointer_cast<op::Constant>(node))
    		{
    			op_annotations->set_cacheable(true);
    			std::cout << "cacheablility is true.\n";
    		}
    		else if (auto parameter = std::dynamic_pointer_cast<op::Parameter>(node))
    		{
    			op_annotations->set_cacheable(parameter->get_cacheable());
    			std::cout << "cacheablility is " << parameter->get_cacheable() << std::endl;
    		}
    		else
    		{
    			bool cacheable = true;
    			for (auto arg : node->get_arguments())
    			{
    				std::cout << "arg is " << arg->get_name() << std::endl;
    				if (auto arg_op = std::dynamic_pointer_cast<op::Op>(arg))
    				{
    					auto arg_op_annotations = arg_op->get_op_annotations();
    					NGRAPH_ASSERT(arg_op_annotations);
    					cacheable = cacheable && arg_op_annotations->is_cacheable();
    				}
    				std::cout << "cacheablility is " << cacheable << std::endl;
    				op_annotations->set_cacheable(cacheable);
    			       if (auto cpu_op_annotations = std::static_pointer_cast<ngraph::runtime::cpu::CPUOpAnnotations>(op_annotations))
    			        {
    			        	std::cout <<"propagage: cpu op annotations\n";
    			        	if (cpu_op_annotations->is_mkldnn_op())
    			        	{
    			        		std::cout << "propagate: use mkldnn\n";
    			        	}
    			        }
    			}
    		}
    		//op->set_op_annotations(op_annotations);
 		}
    }
    std::cout << "done\n";
    return false;
}
