// ----------------------------------------------------------------------------
// copyright 2017 nervana systems inc.
// licensed under the apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
//      http://www.apache.org/licenses/license-2.0
//
// unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// see the license for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            void print_gpu_f32_tensor(void* p, size_t element_count, size_t element_size);
            void check_cuda_errors(CUresult err);
            void cuda_memcpyDtD(void* d, void* s, size_t element_count, size_t element_size);
            void cuda_memcpyHtD(void* d, void* s, size_t buffer_size);
        }
    }
}
