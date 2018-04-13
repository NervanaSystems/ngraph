/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <limits>

#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu;

GPUPrimitiveEmitter::GPUPrimitiveEmitter()
    : m_cuda_emitter(new CUDAEmitter(this))
    , m_cudnn_emitter(new CUDNNEmitter(this))
{
}
GPUPrimitiveEmitter::~GPUPrimitiveEmitter()
{
    for (auto& primitive : m_gpu_primitives)
    {
        delete primitive;
    }
}
std::unique_ptr<CUDAEmitter>& GPUPrimitiveEmitter::get_cuda_emitter()
{
    return m_cuda_emitter;
}
std::unique_ptr<CUDNNEmitter>& GPUPrimitiveEmitter::get_cudnn_emitter()
{
    return m_cudnn_emitter;
}
size_t GPUPrimitiveEmitter::insert(gpu::primitive* f)
{
    m_gpu_primitives.push_back(f);
    return m_gpu_primitives.size() - 1;
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
