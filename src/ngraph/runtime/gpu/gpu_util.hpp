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

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph/runtime/gpu/cuda_error_check.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            void print_gpu_f32_tensor(const void* p, size_t element_count, size_t element_size);
            void check_cuda_errors(CUresult err);
            void* create_gpu_buffer(size_t buffer_size, const void* data = NULL);
            void free_gpu_buffer(void* buffer);
            void cuda_memcpyDtD(void* dst, const void* src, size_t buffer_size);
            void cuda_memcpyHtD(void* dst, const void* src, size_t buffer_size);
            void cuda_memcpyDtH(void* dst, const void* src, size_t buffer_size);
            void cuda_memset(void* dst, int value, size_t buffer_size);
            std::pair<uint64_t, uint64_t> idiv_magic_u32(uint64_t max_numerator, uint64_t divisor);
            std::pair<uint64_t, uint64_t> idiv_magic_u64(uint64_t divisor);
            uint32_t idiv_ceil(int n, int d);

            template <typename T>
            void print_gpu_tensor(const void* p, size_t element_count)
            {
                std::vector<T> local(element_count);
                size_t size_in_bytes = sizeof(T) * element_count;
                cuda_memcpyDtH(local.data(), p, size_in_bytes);
                std::cout << "{" << ngraph::join(local) << "}" << std::endl;
            }

            class StopWatch
            {
            public:
                void start();
                void stop();
                size_t get_call_count();
                size_t get_total_seconds();
                size_t get_total_milliseconds();
                size_t get_total_microseconds();
                size_t get_total_nanoseconds();

            private:
                std::vector<cudaEvent_t> starts;
                std::vector<cudaEvent_t> stops;
                size_t m_total_count = 0;
                size_t m_total_time_in_ns = 0;
                bool m_active = false;
            };

            class StopWatchPool
            {
            public:
                void allocate(size_t num)
                {
                    for (size_t i = 0; i < num; i++)
                    {
                        pool.push_back(StopWatch());
                    }
                }
                StopWatch& get(size_t idx) { return pool[idx]; }
                size_t size() { return pool.size(); }
            private:
                std::vector<StopWatch> pool;
            };
        }
    }
}
