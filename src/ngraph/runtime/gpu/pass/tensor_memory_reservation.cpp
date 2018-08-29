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

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/manager_state.hpp"

#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/gpu/gpu_memory_manager.hpp"
#include "ngraph/runtime/gpu/pass/tensor_memory_reservation.hpp"

using namespace ngraph;

bool ngraph::runtime::gpu::pass::TensorMemoryReservation::run_on_function(
    std::shared_ptr<Function> f)
{
    auto allocator = m_allocator.lock();
    auto buffers = m_memory_buffers.lock();
    if (allocator && buffers)
    {
        size_t mem_pool_size = f->get_temporary_pool_size();
        if (mem_pool_size)
        {
            size_t pool_idx = allocator->reserve_workspace(mem_pool_size, false);
            buffers->insert({f->get_name(), pool_idx});

            return true;
        }
    }
    return false;
}
