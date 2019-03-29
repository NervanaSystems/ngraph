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

if (NGRAPH_INTERPRETER_ENABLE)
    add_library(interpreter_backend SHARED int_backend.cpp node_wrapper.cpp int_executable.cpp)
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(interpreter_backend PROPERTIES
            VERSION ${NGRAPH_VERSION}
            SOVERSION ${NGRAPH_API_VERSION})
    endif()
    target_link_libraries(interpreter_backend PUBLIC ngraph)

    if(NGRAPH_DISTRIBUTED_ENABLE)
        if(NGRAPH_DISTRIBUTED_MLSL_ENABLE)
            target_include_directories(interpreter_backend SYSTEM PRIVATE libmlsl)
            target_link_libraries(interpreter_backend PRIVATE libmlsl)
        elseif(NGRAPH_DISTRIBUTED_OMPI_ENABLE)
            find_package(MPI REQUIRED)
            target_include_directories(interpreter_backend
                SYSTEM PRIVATE ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
            target_link_libraries(interpreter_backend
                PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
        else()
            message(FATAL_ERROR "Distributed Library not supported/mentioned")
        endif()
    endif()

    install(TARGETS interpreter_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
