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
    cuda_emitter.cpp
    cudnn_emitter.cpp
    gpu_backend.cpp
    gpu_call_frame.cpp
    gpu_cuda_context_manager.cpp
    gpu_cuda_function_builder.cpp
    gpu_cuda_function_pool.cpp
    gpu_cuda_kernel_builder.cpp
    gpu_emitter.cpp
    gpu_external_function.cpp
    gpu_invoke.cpp
    gpu_kernel_emitters.cpp
	gpu_memory_manager.cpp
    gpu_primitive_emitter.cpp
    gpu_runtime_context.cpp
    gpu_tensor_view_wrapper.cpp
    gpu_tensor_view.cpp
    gpu_util.cpp
    type_info.cpp
    pass/tensor_memory_reservation.cpp
    )

if (NGRAPH_GPU_ENABLE)
    add_library(gpu_backend SHARED ${SRC})
    set_target_properties(gpu_backend PROPERTIES VERSION ${NGRAPH_VERSION} SOVERSION ${NGRAPH_API_VERSION})
    target_link_libraries(gpu_backend PUBLIC ngraph codegen)
    find_library(CUDA_nvrtc_LIBRARY nvrtc /usr/local/cuda/lib64)
    find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
    find_package(CUDA 8 REQUIRED)
    find_package(CUDNN 7 REQUIRED)
    target_include_directories(gpu_backend SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
    target_link_libraries(gpu_backend PUBLIC
        ${CUDA_cuda_LIBRARY}
        ${CUDA_nvrtc_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDNN_LIBRARIES})
    set_target_properties(gpu_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS gpu_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
