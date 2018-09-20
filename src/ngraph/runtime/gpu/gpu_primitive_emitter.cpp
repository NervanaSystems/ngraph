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

#include <limits>

#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu;

GPUPrimitiveEmitter::GPUPrimitiveEmitter()
    : m_memory_manager(this)
    , m_host_parameters(new GPUHostParameters)
    , m_cuda_emitter(new CUDAEmitter(this, nullptr))
    , m_cudnn_emitter(new CUDNNEmitter(this, nullptr, nullptr))
{
}

GPUPrimitiveEmitter::GPUPrimitiveEmitter(const std::unique_ptr<GPURuntimeContext>& ctx)
    : m_memory_manager(this)
    , m_host_parameters(new GPUHostParameters)
    , m_cuda_emitter(new CUDAEmitter(this, ctx.get()))
    , m_cudnn_emitter(new CUDNNEmitter(this, ctx.get(), this->m_host_parameters))

{
}

std::unique_ptr<CUDAEmitter>& GPUPrimitiveEmitter::get_cuda_emitter()
{
    return m_cuda_emitter;
}
std::unique_ptr<CUDNNEmitter>& GPUPrimitiveEmitter::get_cudnn_emitter()
{
    return m_cudnn_emitter;
}
size_t GPUPrimitiveEmitter::insert(std::unique_ptr<gpu::primitive>&& f)
{
    m_managed_primitives.emplace_back(std::move(f));
    m_gpu_primitives.push_back(m_managed_primitives.back().get());
    return m_gpu_primitives.size() - 1;
}
size_t GPUPrimitiveEmitter::insert(gpu::memory_primitive& f)
{
    m_gpu_mem_primitives.push_back(f);
    return m_gpu_mem_primitives.size() - 1;
}
size_t GPUPrimitiveEmitter::lookup(std::string hash)
{
    if (m_primitive_map.count(hash) > 0)
    {
        return m_primitive_map[hash];
    }
    return std::numeric_limits<size_t>::max();
}
void GPUPrimitiveEmitter::cache(const std::string& hash, const size_t& index)
{
    m_primitive_map.insert({hash, index});
}
