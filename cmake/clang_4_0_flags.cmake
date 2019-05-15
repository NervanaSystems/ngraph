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

set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Werror=return-type")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Werror=inconsistent-missing-override")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Werror=comment")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -pedantic-errors")

 # whitelist errors here
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Weverything")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-c++98-compat-pedantic")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-weak-vtables")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-global-constructors")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-exit-time-destructors")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-missing-prototypes")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-missing-noreturn")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-switch")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-switch-enum")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-covered-switch-default")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-undef")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.1.0)
        set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-zero-as-null-pointer-constant")
    endif()
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.0.0")
        set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-unused-lambda-capture")
    endif()
endif()

# # should remove these
# set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-old-style-cast")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-float-conversion")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-sign-conversion")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-padded")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-sign-compare")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-unused-parameter")

set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-conversion")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-double-promotion")
set(NGRAPH_CLANG_FLAGS "${NGRAPH_CLANG_FLAGS} -Wno-undefined-func-template")
