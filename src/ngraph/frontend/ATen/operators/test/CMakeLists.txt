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

if(NOT NGRAPH_ATEN_TEST_ENABLE)
    message(STATUS "ATen tests disabled")
    return()
endif()

if(NOT NGRAPH_CPU_ENABLE)
    message(STATUS "ATen tests needs CPU enabled")
    return()
endif()

if(NOT NGRAPH_TEST_UTIL_ENABLE)
    message(WARNING "ATen test: Turning on test util!")
    set(NGRAPH_TEST_UTIL_ENABLE ON)
endif()

message(STATUS "fluid tests enabled")

if(LINUX)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.0.0")
        # gtest has issues with this with v1.8.x
        # gtest issue is supposed to be addressed after v1.8.x
        add_compile_options(-Wno-zero-as-null-pointer-constant)
    endif()
endif()

set(SRC
    main.cpp
    dummy.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../../../../../test)

add_executable(ATen-test ${SRC})

target_include_directories(ATen-test PRIVATE "../../../../../../test")

target_link_libraries(ATen-test PRIVATE ngraph_test_util)
target_link_libraries(ATen-test PRIVATE ngraph libgtest)

if(NOT WIN32)
    target_link_libraries(ATen-test PRIVATE pthread)
endif()
target_link_libraries(ATen-test PRIVATE ${CMAKE_DL_LIBS})

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(fluid-test PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()

# So many type_prop tests these days that we need to set /bigobj flag for MSVC.
# We should probably split up type_prop.cpp.
if (MSVC)
    target_compile_options(ATen-test PRIVATE "/bigobj")
endif()

# The INTERPRETER backend is required for convolution, and backwards unit tests
target_link_libraries(ATen-test PRIVATE cpu_backend)
target_link_libraries(ATen-test PRIVATE libmkldnn)
target_compile_definitions(ATen-test PRIVATE NGRAPH_CPU_ENABLE)

if (NGRAPH_TBB_ENABLE)
    target_compile_definitions(ATen-test PRIVATE "NGRAPH_TBB_ENABLE")
endif()
