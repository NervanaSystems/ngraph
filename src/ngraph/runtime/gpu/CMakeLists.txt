# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})

# Add sources for the GPU backend
# and all its dependencies
set(SRC
    gpu_call_frame.cpp
    gpu_backend.cpp
    gpu_emitter.cpp
    gpu_external_function.cpp
    gpu_kernel_emitters.cpp
    gpu_tensor_view.cpp
    gpu_tensor_view_wrapper.cpp
    gpu_util.cpp
    gpu_cuda_kernel_emitters.cpp
    gpu_cuda_kernel_builder.cpp
    gpu_cuda_function_builder.cpp
    gpu_cuda_function_pool.cpp
    gpu_cuda_context_manager.cpp
    gpu_runtime_context.cpp
    gpu_primitive_emitter.cpp
    gpu_invoke.cpp
    cudnn_emitter.cpp
    cuda_emitter.cpp
    type_info.cpp
    )

if (NGRAPH_GPU_ENABLE)
    add_library(gpu_backend SHARED ${SRC})
    target_link_libraries(gpu_backend PUBLIC ngraph codegen)
    find_library(CUDA_nvrtc_LIBRARY nvrtc /usr/local/cuda/lib64)
    find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
    find_package(CUDA 8 REQUIRED)
    find_package(CUDNN 7 REQUIRED)
    target_link_libraries(gpu_backend PUBLIC ${CUDA_cuda_LIBRARY} ${CUDA_nvrtc_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES})
endif()

