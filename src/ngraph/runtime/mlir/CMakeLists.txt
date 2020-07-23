# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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

if (NGRAPH_MLIR_ENABLE OR NGRAPH_CPU_MLIR_ENABLE)
    if (NGRAPH_MLIR_STATIC_LIB_ENABLE)
        set(LIBRARY_TYPE STATIC)
    else()
        set(LIBRARY_TYPE SHARED)
    endif()

    add_library(mlir_backend ${LIBRARY_TYPE} mlir_backend.cpp mlir_executable.cpp)
    target_compile_definitions(mlir_backend PRIVATE MLIR_BACKEND_EXPORTS)
    if(NGRAPH_LIB_VERSIONING_ENABLE)
        set_target_properties(mlir_backend PROPERTIES
            VERSION ${NGRAPH_VERSION}
            SOVERSION ${NGRAPH_API_VERSION})
    endif()
    target_link_libraries(mlir_backend PUBLIC ngraph mlir_llvm_backend)
    if(NGRAPH_MLIR_STATIC_LIB_ENABLE)
        target_compile_definitions(mlir_backend PUBLIC NGRAPH_MLIR_STATIC_LIB_ENABLE)
    endif()

    # TODO: can we get away without LLVM/MLIR include path.
    # Currently mlir backend compiler.hpp include LLVM/MLIR files
    get_directory_property(MLIR_LLVM_INCLUDEPATH
                            DIRECTORY ${NGRAPH_MLIR_SOURCE_DIR}
                            DEFINITION MLIR_LLVM_INCLUDEPATH)
    target_include_directories(mlir_backend PUBLIC ${MLIR_INCLUDE_PATHS} ${MLIR_LLVM_INCLUDE_PATH})

    install(TARGETS mlir_backend
        LIBRARY DESTINATION "${NGRAPH_INSTALL_LIB}"
        ARCHIVE DESTINATION "${NGRAPH_INSTALL_LIB}"
        RUNTIME DESTINATION "${NGRAPH_INSTALL_LIB}"
    )
endif()
