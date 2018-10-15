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

bool ngraph::runtime::gpu::pass::KernelMemoryAllocation::run_on_function(
    std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (auto bnbp = dynamic_pointer_cast<op::BatchNormBackprop>(n))
        {
            auto bn0 = dynamic_pointer_cast<op::BatchNorm>(bnbp->get_argument(3));
            auto bn1 = dynamic_pointer_cast<op::BatchNorm>(bnbp->get_argument(4));
            if (bn0 && bn0 == bn1)
            {
                //std::make_shared<op::gpu::BatchNorm>(bn0)
                replace = true;
            }
        }
    }
    return replaced;
}
