# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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

set (SRC
    nbench.cpp
    benchmark.cpp
)

add_executable(nbench ${SRC})

if (APPLE)
    set_property(TARGET nbench APPEND_STRING PROPERTY LINK_FLAGS " -Wl,-rpath,@loader_path/../lib")
endif()
target_link_libraries(nbench PRIVATE ngraph)
if (NGRAPH_CPU_ENABLE)
    target_link_libraries(nbench PRIVATE cpu_backend)
endif()
if (NGRAPH_INTELGPU_ENABLE)
    target_link_libraries(nbench PRIVATE intelgpu_backend)
endif()
if (NGRAPH_GPU_ENABLE)
    target_link_libraries(nbench PRIVATE gpu_backend)
endif()
if (NGRAPH_INTERPRETER_ENABLE)
    target_link_libraries(nbench PRIVATE interpreter_backend)
endif()
if (NGRAPH_PLAIDML_ENABLE)
    target_link_libraries(nbench PRIVATE plaidml_backend)
endif()
if (NGRAPH_GENERIC_CPU_ENABLE)
    target_link_libraries(nbench PRIVATE gcpu_backend)
endif()

if(NGRAPH_DISTRIBUTED_ENABLE)
    if(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
        target_link_libraries(nbench PRIVATE libmlsl)
    elseif(NGRAPH_DISTRIBUTED_OMPI_ENABLE)
        find_package(MPI REQUIRED)
        target_include_directories(nbench SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
        target_link_libraries(nbench  PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    else()
            message(FATAL_ERROR "Distributed Library not supported/mentioned")
    endif()
endif()

install(TARGETS nbench RUNTIME DESTINATION ${NGRAPH_INSTALL_BIN})
