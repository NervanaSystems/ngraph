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

#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/nvshape.hpp"

using namespace ngraph;

TEST(gpu_test, gpu_shape_from_64bit_shape)
{
    Shape shape{1UL << 33};
    ASSERT_ANY_THROW([](NVShape s) {}(shape););
}

TEST(gpu_test, memory_manager_unallocated)
{
    runtime::gpu::GPUPrimitiveEmitter emitter;
    auto allocator = emitter.get_memory_allocator();
    size_t idx = allocator.reserve_workspace(10);
    runtime::gpu::memory_primitive& mem_primitive = emitter.get_memory_primitives()[idx];
    ASSERT_ANY_THROW(mem_primitive());
}

TEST(gpu_test, memory_manager_allocated)
{
    runtime::gpu::GPUPrimitiveEmitter emitter;
    size_t idx;
    {
        auto allocator = emitter.get_memory_allocator();
        idx = allocator.reserve_workspace(10);
    }
    emitter.allocate_primitive_memory();
    runtime::gpu::memory_primitive& mem_primitive = emitter.get_memory_primitives()[idx];
    EXPECT_NO_THROW(mem_primitive());
}

TEST(gpu_test, memory_manager_extract_arguments)
{
    std::vector<float> fp32_args = {2112.0f, 2112.0f};
    runtime::gpu::GPUPrimitiveEmitter emitter;
    size_t idx;
    {
        auto allocator = emitter.get_memory_allocator();
        idx = allocator.reserve_argspace(fp32_args.data(), fp32_args.size() * sizeof(float));
    }
    emitter.allocate_primitive_memory();
    runtime::gpu::memory_primitive& mem_primitive = emitter.get_memory_primitives()[idx];
    std::vector<float> host(2, 0);
    runtime::gpu::cuda_memcpyDtH(host.data(), mem_primitive(), host.size() * sizeof(float));
    EXPECT_EQ(host, fp32_args);
}

TEST(gpu_test, memory_manager_argspace_size)
{
    std::vector<float> fp32_args = {2112.0f, 2112.0f};
    runtime::gpu::GPUPrimitiveEmitter emitter;
    {
        auto allocator = emitter.get_memory_allocator();
        allocator.reserve_argspace(fp32_args.data(), fp32_args.size() * sizeof(float));
    }
    emitter.allocate_primitive_memory();
    EXPECT_EQ(emitter.sizeof_device_allocation(), fp32_args.size() * sizeof(float));
}

TEST(gpu_test, memory_manager_overlapping_workspace_allocsize)
{
    runtime::gpu::GPUPrimitiveEmitter emitter;
    for (size_t i = 0; i < 8; i++)
    {
        auto allocator = emitter.get_memory_allocator();
        allocator.reserve_workspace(std::pow(2, i));
    }
    emitter.allocate_primitive_memory();
    EXPECT_EQ(emitter.sizeof_device_allocation(), 128);

    void* first = nullptr;
    for (size_t i = 0; i < 8; i++)
    {
        if (not first)
        {
            first = emitter.get_memory_primitives()[i]();
        }
        else
        {
            EXPECT_EQ(emitter.get_memory_primitives()[i](), first);
        }
    }
}

TEST(gpu_test, memory_manager_seperate_workspaces_allocsize)
{
    size_t total_size = 0;
    runtime::gpu::GPUPrimitiveEmitter emitter;
    {
        auto allocator = emitter.get_memory_allocator();
        for (size_t i = 0; i < 8; i++)
        {
            size_t size = std::pow(2, i);
            allocator.reserve_workspace(size);
            total_size += pass::MemoryManager::align(size, 8);
        }
    }
    emitter.allocate_primitive_memory();
    EXPECT_EQ(emitter.sizeof_device_allocation(), total_size);
}
