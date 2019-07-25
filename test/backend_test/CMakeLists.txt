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
    backend_test.cpp
    main.cpp
    op_slice.cpp
)

add_executable(backend-test ${SRC})

target_include_directories(backend-test PRIVATE "..")

if(NGRAPH_ADDRESS_SANITIZER)
    add_compile_options(-g -fsanitize=address -fno-omit-frame-pointer)
endif()

target_link_libraries(backend-test PRIVATE ngraph_test_util)
target_link_libraries(backend-test PRIVATE ngraph libgtest)
if(NOT WIN32)
    target_link_libraries(backend-test PRIVATE pthread)
endif()
target_link_libraries(backend-test PRIVATE ${CMAKE_DL_LIBS})

# FIXME: This is a false dependency, but for some reason the link line gets
# messed up without it.
if (NGRAPH_INTERPRETER_ENABLE)
    target_compile_definitions(backend-test PRIVATE NGRAPH_INTERPRETER_ENABLE)
    target_link_libraries(backend-test PRIVATE interpreter_backend)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
    target_compile_options(backend-test PRIVATE -Wno-undef -Wno-reserved-id-macro)
endif()
