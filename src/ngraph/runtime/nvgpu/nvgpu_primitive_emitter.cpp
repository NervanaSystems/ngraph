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

#include <limits>

#include "ngraph/runtime/nvgpu/nvgpu_primitive_emitter.hpp"

using namespace ngraph;
using namespace ngraph::runtime::nvgpu;

NVPrimitiveEmitter::NVPrimitiveEmitter()
    : m_memory_manager(this)
    , m_host_parameters(new NVHostParameters)
    , m_host_emitter(new HostEmitter(this, nullptr))
    , m_cuda_emitter(new CUDAEmitter(this, nullptr, nullptr))
    , m_cudnn_emitter(new CUDNNEmitter(this, nullptr, nullptr))
    , m_cublas_emitter(new CUBLASEmitter(this, nullptr))
{
}

NVPrimitiveEmitter::NVPrimitiveEmitter(const std::unique_ptr<NVRuntimeContext>& ctx)
    : m_memory_manager(this)
    , m_host_parameters(new NVHostParameters)
    , m_host_emitter(new HostEmitter(this, ctx.get()))
    , m_cuda_emitter(new CUDAEmitter(this, ctx.get(), this->m_host_parameters))
    , m_cudnn_emitter(new CUDNNEmitter(this, ctx.get(), this->m_host_parameters))
    , m_cublas_emitter(new CUBLASEmitter(this, ctx.get()))

{
}

std::unique_ptr<HostEmitter>& NVPrimitiveEmitter::get_host_emitter()
{
    return m_host_emitter;
}
std::unique_ptr<CUDAEmitter>& NVPrimitiveEmitter::get_cuda_emitter()
{
    return m_cuda_emitter;
}
std::unique_ptr<CUDNNEmitter>& NVPrimitiveEmitter::get_cudnn_emitter()
{
    return m_cudnn_emitter;
}
std::unique_ptr<CUBLASEmitter>& NVPrimitiveEmitter::get_cublas_emitter()
{
    return m_cublas_emitter;
}
size_t NVPrimitiveEmitter::insert(std::unique_ptr<nvgpu::primitive>&& f)
{
    m_managed_primitives.emplace_back(std::move(f));
    m_nvgpu_primitives.push_back(m_managed_primitives.back().get());
    return m_nvgpu_primitives.size() - 1;
}
size_t NVPrimitiveEmitter::insert(const nvgpu::memory_primitive& f)
{
    m_nvgpu_mem_primitives.push_back(f);
    return m_nvgpu_mem_primitives.size() - 1;
}
size_t NVPrimitiveEmitter::lookup(const std::string& hash)
{
    auto it = m_primitive_map.find(hash);
    if (it != m_primitive_map.end())
    {
        return it->second;
    }
    return std::numeric_limits<size_t>::max();
}
void NVPrimitiveEmitter::cache(const std::string& hash, const size_t& index)
{
    m_primitive_map.insert({hash, index});
}

size_t NVPrimitiveEmitter::register_primitive(std::unique_ptr<nvgpu::primitive>& f, std::string hash)
{
    size_t primitive_index = this->insert(std::move(f));
    this->cache(hash, primitive_index);
    return primitive_index;
}
