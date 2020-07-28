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

add_compile_options(-Werror=return-type)
add_compile_options(-Werror=inconsistent-missing-override)
add_compile_options(-Werror=comment)
add_compile_options(-pedantic-errors)
if(NGRAPH_STRICT_ERROR_CHECK)
    # These flags are for clang-8
    add_compile_options(-Weverything)

    # whitelist errors here
    add_compile_options(-Wno-c++98-compat-pedantic)
    add_compile_options(-Wno-covered-switch-default)
    add_compile_options(-Wno-deprecated)
    add_compile_options(-Wno-documentation-unknown-command)
    add_compile_options(-Wno-documentation)
    add_compile_options(-Wno-double-promotion)
    add_compile_options(-Wno-exit-time-destructors)
    add_compile_options(-Wno-extra-semi-stmt)
    add_compile_options(-Wno-extra-semi)
    add_compile_options(-Wno-float-conversion)
    add_compile_options(-Wno-float-equal)
    add_compile_options(-Wno-global-constructors)
    add_compile_options(-Wno-implicit-fallthrough)
    add_compile_options(-Wno-implicit-float-conversion)
    add_compile_options(-Wno-implicit-int-conversion) # needed
    add_compile_options(-Wno-missing-prototypes)
    add_compile_options(-Wno-newline-eof)
    add_compile_options(-Wno-padded)
    add_compile_options(-Wno-return-std-move-in-c++11)
    add_compile_options(-Wno-shadow)
    add_compile_options(-Wno-shorten-64-to-32) # needed
    add_compile_options(-Wno-sign-compare) # needed
    add_compile_options(-Wno-sign-conversion) # needed
    add_compile_options(-Wno-switch-enum)
    add_compile_options(-Wno-undefined-func-template)
    add_compile_options(-Wno-unused-exception-parameter)
    add_compile_options(-Wno-unused-parameter)
    add_compile_options(-Wno-unused-template)
    add_compile_options(-Wno-weak-vtables)
else()
    add_compile_options(-Wall)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.1.0)
        #add_compile_options(-Wno-zero-as-null-pointer-constant)
    endif()
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(AppleClang)?Clang$")
    add_compile_options(-Wno-c99-extensions)
endif()
