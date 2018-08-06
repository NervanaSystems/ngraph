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

#pragma once

#include <memory>
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPUAllocator;
            namespace pass
            {
                class TensorMemoryReservation;
            }
        }
    }
}

class ngraph::runtime::gpu::pass::TensorMemoryReservation : public ngraph::pass::FunctionPass
{
public:
    TensorMemoryReservation(std::weak_ptr<ngraph::runtime::gpu::GPUAllocator> allocator,
                            std::weak_ptr<std::unordered_map<std::string, size_t>> buffers)
        : ngraph::pass::FunctionPass()
        , m_allocator(allocator)
        , m_memory_buffers(buffers)
    {
    }

    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f);

private:
    std::weak_ptr<ngraph::runtime::gpu::GPUAllocator> m_allocator;
    std::weak_ptr<std::unordered_map<std::string, size_t>> m_memory_buffers;
};
