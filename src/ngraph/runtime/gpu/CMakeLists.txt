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
    cublas_emitter.cpp
    gpu_backend.cpp
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
    gpu_tensor_wrapper.cpp
    gpu_tensor.cpp
    gpu_util.cpp
    type_info.cpp
    pass/gpu_layout.cpp
    pass/tensor_memory_reservation.cpp
    gpu_kernel_args.cpp
    pass/gpu_rnn_fusion.cpp
    pass/gpu_batch_norm_cache.cpp
    op/batch_norm.cpp
    op/rnn.cpp
    )
set(CUDA_INC
    ${PROJECT_SOURCE_DIR}/src/
    )
set(CUDA_SRC
    nvcc/example.cu.cpp
    )

if (NGRAPH_GPU_ENABLE)
    find_package(CUDA 9 QUIET)
    if (CUDA_FOUND)
        set(CUDA9_FOUND TRUE)
        message(STATUS "Found CUDA 9")
    else()
        find_package(CUDA 8 REQUIRED)
    endif()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        --compiler-options -fPIC;
        -arch=sm_30;
        -gencode=arch=compute_35,code=sm_35;
        -gencode=arch=compute_50,code=sm_50;
        -gencode=arch=compute_52,code=sm_52;
        -gencode=arch=compute_60,code=sm_60;
        -gencode=arch=compute_61,code=sm_61;
        -gencode=arch=compute_61,code=compute_61)
    if (CUDA9_FOUND)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
            -gencode=arch=compute_62,code=sm_62;
            -gencode=arch=compute_70,code=sm_70;
            -gencode=arch=compute_70,code=compute_70)
    endif()

    set (DO_CUDA_COMPILE FALSE)
    if (CUDA9_FOUND)
        if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
            # CUDA 9 supports up to gcc 6.x
            if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
                set (DO_CUDA_COMPILE TRUE)
            else()
                message(STATUS "NVCC will not be used because CUDA 9 only supports up to gcc 6.x")
            endif()
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            # CUDA 9 supports up to clang 3.9
            if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.0)
                set (DO_CUDA_COMPILE TRUE)
            else()
                message(STATUS "NVCC will not be used because CUDA 9 only supports up to clang 3.9")
            endif()
        endif()
    else()
        # CUDA 8 (minimum version of CUDA we support)
        if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
            # Current release of CUDA 8 supports up to gcc 5.4
            if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.5)
                set (DO_CUDA_COMPILE TRUE)
            else()
                message(STATUS "NVCC will not be used because CUDA 8 only supports up to gcc 5.4")
            endif()
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            # CUDA 8 supports up to clang 3.8
            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.9)
                set (DO_CUDA_COMPILE TRUE)
            else()
                message(STATUS "NVCC will not be used because CUDA 8 only supports up to clang 3.8")
            endif()
        endif()
    endif()

    if (DO_CUDA_COMPILE)
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            # CUDA_PROPAGATE_HOST_FLAGS is true by default, so disable
            # clang warnings that are known to flag CUDA code
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                --compiler-options -Wno-reserved-id-macro;
                --compiler-options -Wno-undef;
                --compiler-options -Wno-old-style-cast;
                --compiler-options -Wno-deprecated;
                --compiler-options -Wno-unused-macros;
                --compiler-options -Wno-used-but-marked-unused)
        endif()

        message(STATUS "Precompiling static CUDA kernels via NVCC")
        set_source_files_properties( ${CUDA_SRC} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
        cuda_include_directories(${CUDA_INC})
        cuda_compile(CUDA_OBJ ${CUDA_SRC} STATIC)
    else()
        message(STATUS "Not precompiling static CUDA kernels via NVCC; runtime compilation via NVRTC will be used.")
    endif()

    add_library(gpu_backend SHARED ${SRC} ${CUDA_OBJ})
    set_target_properties(gpu_backend PROPERTIES VERSION ${NGRAPH_VERSION} SOVERSION ${NGRAPH_API_VERSION})
    target_link_libraries(gpu_backend PUBLIC ngraph codegen)
    find_library(CUDA_nvrtc_LIBRARY nvrtc /usr/local/cuda/lib64)
    find_library(CUDA_cuda_LIBRARY cuda /usr/local/cuda/lib64/stubs)
    find_library(CUDA_cudart_LIBRARY libcudart_static.a /usr/local/cuda/lib64)

    find_package(CUDNN 7 REQUIRED)
    target_include_directories(gpu_backend SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR})
    target_link_libraries(gpu_backend PUBLIC
        ${CUDA_cuda_LIBRARY}
        ${CUDA_nvrtc_LIBRARY}
        ${CUDA_cudart_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDNN_LIBRARIES})
    set_target_properties(gpu_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})

    install(TARGETS gpu_backend LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB})
endif()
