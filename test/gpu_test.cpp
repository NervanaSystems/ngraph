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

#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/nvshape.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/random.hpp"

using namespace std;
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

TEST(gpu_test, memory_manager_zero_workspace)
{
    runtime::gpu::GPUPrimitiveEmitter emitter;
    size_t idx_null, idx_not_null;
    {
        auto allocator = emitter.get_memory_allocator();
        idx_null = allocator.reserve_workspace(0);
        idx_not_null = allocator.reserve_workspace(10);
    }
    emitter.allocate_primitive_memory();
    EXPECT_EQ(emitter.get_memory_primitives()[idx_null](), nullptr);
    EXPECT_NE(emitter.get_memory_primitives()[idx_not_null](), nullptr);
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

// This test is add to catch a potential bug in allocator
// previously allocator will copy extra data
// for exampele: alignment = 8 bytes, you reserve 4 bytes space
// previously allocator will copy 8 bytes data from input_args, this will lead to two potential bug:
// 1. copy extrea data intead of initial alignment data to 0.
// 2. out of boundary access for input_args which lead to undefined behavior
TEST(gpu_test, memory_manager_argspace_alignment)
{
    size_t alignment = 8;
    std::vector<char> input_args = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<char> ref_args = {0, 1, 2, 3, 0, 0, 0, 0};
    std::vector<char> result_args(alignment, 0);
    size_t idx;
    runtime::gpu::GPUPrimitiveEmitter emitter;
    {
        auto allocator = emitter.get_memory_allocator();
        idx = allocator.reserve_argspace(input_args.data(), 4 * sizeof(char));
    }
    emitter.allocate_primitive_memory();
    runtime::gpu::memory_primitive& mem_primitive = emitter.get_memory_primitives()[idx];
    runtime::gpu::cuda_memcpyDtH(result_args.data(), mem_primitive(), alignment * sizeof(char));
    EXPECT_EQ(result_args, ref_args);
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

TEST(gpu_test, topk_fanout_graph_transform)
{
    Shape shape{2, 3, 2};
    Shape out_shape{2, 2, 2};
    auto A_gpu = make_shared<op::Parameter>(element::f32, shape);
    auto A_int32_gpu_1 = make_shared<op::Parameter>(element::i32, out_shape);
    auto A_int32_gpu_2 = make_shared<op::Parameter>(element::i32, out_shape);
    auto A_f32_gpu_1 = make_shared<op::Parameter>(element::f32, out_shape);
    auto A_f32_gpu_2 = make_shared<op::Parameter>(element::f32, out_shape);
    auto B_gpu = make_shared<op::TopK>(A_gpu, 1, element::i32, 2, true);
    auto C_gpu_0 = make_shared<op::GetOutputElement>(B_gpu, 0);
    auto C_gpu_1 = make_shared<op::GetOutputElement>(B_gpu, 1);

    auto gpu_R_0 = make_shared<op::Add>(A_int32_gpu_1, C_gpu_0);
    auto gpu_R_1 = make_shared<op::Add>(A_int32_gpu_2, C_gpu_0);
    auto gpu_R_2 = make_shared<op::Add>(A_f32_gpu_1, C_gpu_1);
    auto gpu_R_3 = make_shared<op::Add>(A_f32_gpu_2, C_gpu_1);

    auto gpu_f = make_shared<Function>(
        NodeVector{gpu_R_0, gpu_R_1, gpu_R_2, gpu_R_3},
        ParameterVector{A_gpu, A_int32_gpu_1, A_int32_gpu_2, A_f32_gpu_1, A_f32_gpu_2});

    auto backend = runtime::Backend::create("GPU");

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(
        a, vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 3.0f, 3.0f, 1.0f, 4.0f});
    auto b = backend->create_tensor(element::i32, out_shape);
    copy_data(b, vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0});
    auto c = backend->create_tensor(element::i32, out_shape);
    copy_data(c, vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0});
    auto d = backend->create_tensor(element::f32, out_shape);
    copy_data(d, vector<float>{0, 0, 0, 0, 0, 0, 0, 0});
    auto e = backend->create_tensor(element::f32, out_shape);
    copy_data(e, vector<float>{0, 0, 0, 0, 0, 0, 0, 0});

    auto r0 = backend->create_tensor(element::i32, out_shape);
    auto r1 = backend->create_tensor(element::i32, out_shape);
    auto r2 = backend->create_tensor(element::f32, out_shape);
    auto r3 = backend->create_tensor(element::f32, out_shape);

    auto handle = backend->compile(gpu_f);
    backend->call_with_validate(handle, {r0, r1, r2, r3}, {a, b, c, d, e});

    EXPECT_EQ((vector<int32_t>{2, 1, 1, 2, 1, 2, 0, 1}), read_vector<int32_t>(r0));
    EXPECT_EQ((vector<int32_t>{2, 1, 1, 2, 1, 2, 0, 1}), read_vector<int32_t>(r1));
    EXPECT_TRUE(
        test::all_close_f(vector<float>{4, 4, 3, 3, 3, 4, 2, 3}, read_vector<float>(r2), 24, 0));
    EXPECT_TRUE(
        test::all_close_f(vector<float>{4, 4, 3, 3, 3, 4, 2, 3}, read_vector<float>(r3), 24, 0));
    auto reshape_count = count_ops_of_type<ngraph::op::Reshape>(gpu_f);
    EXPECT_EQ(reshape_count, 10);
}

//
// This test primarly checks that maxpool backprop functions
// correctly when the input tensor is larger than most cache sizes.
// Here the to-be-pooled tensor is rank 2 with one non-trivial
// dimension:
//
// x : [[0, 1, 0, 1, 0, 1, ... , 0, 1]]  <--- input data
//       ----  ----  ----  ...   ----    <--- pooling windows
// y : [[ 1  ,  1  ,  1  , ... ,  1]]    <--- max pooled output
//
// The pooling window is size 2 and stride 2, so the windows
// do not overlap. Thus, each window will effectively see [0, 1]
// as its input data for max pooling. The resulting output tensor
// of pooling will be sizeof(x) with all elements equal to 1 as
// seen above.
// Therefore, for the backward pooling operation with the same window shape
// and strides, the value of dy will only propogate to the positions in
// dx that correspond to a value of 1 in the corresponding input tensor x:
//
// dy : [[2, 3, ... , 4]]
// x  : [[0, 1, 0, 1, ... , 0, 1]]
// dx : [[0, 2, 0, 3, ... , 0, 4]]
//
TEST(gpu_test, maxpool_bprop_larger_than_cache)
{
    Shape window_shape{1, 2};
    Strides move_strides{1, 2};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};

    // 200 MB tensor to exceed cache
    const size_t num_elements = 50 * 1024 * 1024;
    auto ceil_div = [](size_t x, size_t y) { return 1 + ((x - 1) / y); };
    const size_t num_pooled_elements = ceil_div(num_elements + padding_below.back() +
                                                    padding_above.back() - window_shape.back() + 1,
                                                move_strides.back());
    Shape shape_x{1, 1, 1, num_elements};
    Shape shape_y{1, 1, 1, num_pooled_elements};

    auto x = make_shared<op::Parameter>(element::f32, shape_x);
    auto dy = make_shared<op::Parameter>(element::f32, shape_y);
    auto bprop =
        make_shared<Function>(make_shared<op::MaxPoolBackprop>(
                                  x, dy, window_shape, move_strides, padding_below, padding_above),
                              ParameterVector{x, dy});

    auto backend = runtime::Backend::create("GPU");

    // initialize x to array of alternating 0s and 1s as described above
    std::vector<float> x_data(num_elements, 0);
    for (auto i = 0u; i < num_elements; i++)
    {
        x_data[i] = (i % 2);
    }
    auto x_t = backend->create_tensor(element::f32, shape_x);
    copy_data(x_t, x_data);

    // use random data for deltas dy
    std::vector<float> dy_data(num_pooled_elements);
    test::Uniform<float> rng(0.0f, 1.0f);
    rng.initialize(dy_data);
    auto dy_t = backend->create_tensor(element::f32, shape_y);
    copy_data(dy_t, dy_data);

    // create result deltas tensor and run the backward max pooling operation
    auto dx_t = backend->create_tensor(element::f32, shape_x);
    auto handle = backend->compile(bprop);
    backend->call_with_validate(handle, {dx_t}, {x_t, dy_t});

    // expected values should be dy with 0s left inserted
    // for each delta, see test description above for details
    std::vector<float> expected_dx(num_elements, 0);
    for (auto i = 0u, j = 0u; i < num_elements; i++)
    {
        if (x_data[i])
        {
            expected_dx[i] = x_data[i] * dy_data[j++];
        }
    }
    EXPECT_EQ(expected_dx, read_vector<float>(dx_t));
}
