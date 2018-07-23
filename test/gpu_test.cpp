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
#include "ngraph/runtime/gpu/gpu_shape.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/gpu/op/memory_wrapped_node.hpp"
#include "ngraph/runtime/gpu/pass/kernel_memory_allocation.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

TEST(gpu_test, gpu_shape_from_64bit_shape)
{
    Shape shape{1UL << 33};
    ASSERT_ANY_THROW([](GPUShape s) {}(shape););
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
    auto allocator = emitter.get_memory_allocator();
    size_t idx = allocator.reserve_workspace(10);
    emitter.allocate_primitive_memory();
    runtime::gpu::memory_primitive& mem_primitive = emitter.get_memory_primitives()[idx];
    EXPECT_NO_THROW(mem_primitive());
}

TEST(gpu_test, memory_manager_extract_arguments)
{
    runtime::gpu::GPUPrimitiveEmitter emitter;
    auto allocator = emitter.get_memory_allocator();
    std::vector<float> fp32_args = {2112.0f, 2112.0f};
    size_t idx = allocator.reserve_argspace(fp32_args.data(), fp32_args.size() * sizeof(float));
    emitter.allocate_primitive_memory();
    runtime::gpu::memory_primitive& mem_primitive = emitter.get_memory_primitives()[idx];
    std::vector<float> host(2, 0);
    runtime::gpu::cuda_memcpyDtH(host.data(), mem_primitive(), host.size() * sizeof(float));
    EXPECT_EQ(host, fp32_args);
}

TEST(gpu_test, memory_manager_argspace_size)
{
    runtime::gpu::GPUPrimitiveEmitter emitter;
    auto allocator = emitter.get_memory_allocator();
    std::vector<float> fp32_args = {2112.0f, 2112.0f};
    allocator.reserve_argspace(fp32_args.data(), fp32_args.size() * sizeof(float));
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


TEST(gpu_test, memory_wrapped_node_bprop)
{
    Shape shape{2, 3};
    auto data = std::make_shared<op::Parameter>(element::f32, shape);
    auto delta = std::make_shared<op::Parameter>(element::f32, shape);
    auto softmax = std::make_shared<op::Softmax>(data, AxisSet{1});

    pass::Manager pass_manager;
    pass_manager.register_pass<runtime::gpu::pass::KernelMemoryAllocation>();
    pass_manager.register_pass<pass::VisualizeTree>("wrapped_softmax");
    auto f = std::make_shared<Function>(softmax, op::ParameterVector{data});

    ngraph::autodiff::Adjoints adjoints(NodeVector{softmax}, NodeVector{delta});

    auto d_data = adjoints.backprop_node(data);

    auto df = std::make_shared<Function>(NodeVector{d_data}, op::ParameterVector{data, delta});
    pass_manager.run_passes(df);


    EXPECT_EQ(0,0);
}
