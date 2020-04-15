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

#include <memory>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/manager_state.hpp"
#include "ngraph/runtime/gpu/gpu_memory_manager.hpp"
#include "ngraph/runtime/gpu/pass/tensor_memory_reservation.hpp"

using namespace ngraph;
using namespace std;

bool runtime::gpu::pass::TensorMemoryReservation::run_on_function(shared_ptr<Function> f)
{
    bool reservation = false;
    size_t mem_pool_size = f->get_temporary_pool_size();
    // intermediate memory reservation
    if (mem_pool_size)
    {
        size_t pool_idx = m_allocator.reserve_workspace(mem_pool_size, false);
        m_memory_buffers.insert({f->get_name(), pool_idx});
        reservation = true;
    }

    // constant memory reservation
    for (auto const& node : f->get_ops())
    {
        if (auto constant = std::dynamic_pointer_cast<ngraph::op::Constant>(node))
        {
            std::shared_ptr<descriptor::Tensor> tv = node->get_outputs()[0].get_tensor_ptr();
            size_t idx = m_allocator.reserve_argspace(constant->get_data_ptr(), tv->size());
            m_memory_buffers.insert({node->get_name(), idx});
            reservation = true;
        }
    }

    return reservation;
}
