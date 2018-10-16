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

#include <memory>
#include <unordered_map>

#include "ngraph/runtime/gpu/op/batch_norm.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/runtime/gpu/pass/gpu_batch_norm_cache.hpp"

using namespace ngraph;

#define RETURN_IF_FALSE(cond, message)          \
    if (!(cond))                                \
    {                                           \
        NGRAPH_DEBUG << message;                \
        return false;                           \
    }

bool ngraph::runtime::gpu::pass::BatchNormCache::run_on_function(
    std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (auto bnbp = std::dynamic_pointer_cast<op::BatchNormBackprop>(n))
        {
            // pass must be run prior to GOE elimination
            // auto input_goe = std::dynamic_pointer_cast<op::GetOutputElement>(bnbp->get_argument(2));
            // auto mean_goe = std::dynamic_pointer_cast<op::GetOutputElement>(bnbp->get_argument(3));
            // auto var_goe = std::dynamic_pointer_cast<op::GetOutputElement>(bnbp->get_argument(4));
            std::vector<std::shared_ptr<op::GetOutputElement> > goes;
            for (auto& arg : bnbp->get_arguments())
            {
                if (auto goe = std::dynamic_pointer_cast<op::GetOutputElement>(arg))
                {
                    for (auto& target : goe->get_arguments())
                    {
                        if (auto bn = std::dynamic_pointer_cast<op::BatchNorm>(target))
                        {
                            goes.push_back(goe);
                        }
                    }
                }
            }

            if (goes.size())
            {

                if (auto target = std::dynamic_pointer_cast<op::BatchNorm>(goes.front()->get_arguments().at(0)))
                {
                    auto replacement = std::make_shared<op::gpu::CUDNNBatchNorm>(target->get_eps_value(),
                                                                                 target->get_argument(0),
                                                                                 target->get_argument(1),
                                                                                 target->get_argument(2));
                    for (auto& goe : goes)
                    {
                        auto new_goe = std::make_shared<op::GetOutputElement>(replacement, goe->get_n());
                        ngraph::replace_node(goe, new_goe);
                        replaced = true;
                    }
                }
            }
        }
    }
    return replaced;
}
