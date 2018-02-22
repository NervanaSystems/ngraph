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

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            void print_gpu_f32_tensor(void* p, size_t element_count, size_t element_size);
            void check_cuda_errors(CUresult err);
            void* create_gpu_buffer(size_t buffer_size);
            void cuda_memcpyDtD(void* d, void* s, size_t element_count, size_t element_size);
            void cuda_memcpyHtD(void* d, void* s, size_t buffer_size);
        }
    }
}
