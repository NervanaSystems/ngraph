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

# Enable ExternalProject CMake module
include(ExternalProject)

#------------------------------------------------------------------------------
# Download and install GoogleTest ...
#------------------------------------------------------------------------------

set(CLDNN_GIT_REPO_URL https://github.com/intel/clDNN.git)
set(CLDNN_GIT_LABEL a05c9f29c5690a4beb9cd8c9f712cfb4399b2a6d)
set(BOOST_VERSION 1.64.0)
set(OUT_DIR ${EXTERNAL_PROJECTS_ROOT}/cldnn/out)

if(NOT WIN32)
    set(COMPILE_FLAGS -fPIC)
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (DEFINED NGRAPH_USE_CXX_ABI)
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI}")
    endif()
endif()

ExternalProject_Add(
    ext_cldnn
    PREFIX cldnn
    GIT_REPOSITORY ${CLDNN_GIT_REPO_URL}
    GIT_TAG ${CLDNN_GIT_LABEL}
    # Disable install step
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_GENERATOR ${CMAKE_GENERATOR}
    CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
    CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
    CMAKE_ARGS
                ${NGRAPH_FORWARD_CMAKE_ARGS}
                # -DCLDNN__OUTPUT_DIR=out/Debug
                -DCLDNN__BOOST_VERSION=${BOOST_VERSION}
                -DCLDNN__INCLUDE_TESTS=FALSE
                -DCLDNN__INCLUDE_TUTORIAL=FALSE
    EXCLUDE_FROM_ALL TRUE
    )

#------------------------------------------------------------------------------

add_library(libcldnn INTERFACE)
if (CLDNN_ROOT_DIR)
    find_package(CLDNN REQUIRED)
    target_include_directories(libcldnn SYSTEM INTERFACE ${CLDNN_INCLUDE_DIRS})
    target_link_libraries(libcldnn INTERFACE ${CLDNN_LIBRARIES})
    install(
	FILES 
	    ${CLDNN_LIBRARIES}
	DESTINATION 
	    ${NGRAPH_INSTALL_LIB}
	OPTIONAL
        )
else()
    ExternalProject_Get_Property(ext_cldnn SOURCE_DIR BINARY_DIR)
    set(CLDNN_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}clDNN64${CMAKE_SHARED_LIBRARY_SUFFIX})
    ExternalProject_Add_Step(
        ext_cldnn
        CopyCLDNN
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_DIR}/build/out/Linux64/${CMAKE_BUILD_TYPE}/${CLDNN_LIB} ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${CLDNN_LIB}
        COMMENT "Copy cldnn runtime libraries to ngraph build directory."
        DEPENDEES install
        )
    add_dependencies(libcldnn ext_cldnn)
    target_include_directories(libcldnn SYSTEM INTERFACE ${SOURCE_DIR}/api)
    target_link_libraries(libcldnn INTERFACE ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${CLDNN_LIB})
    install(
        FILES
            ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${CLDNN_LIB}
        DESTINATION
            ${NGRAPH_INSTALL_LIB}
        OPTIONAL
        )
endif()
