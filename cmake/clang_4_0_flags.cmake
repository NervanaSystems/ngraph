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

add_compile_options(-Werror=return-type)
add_compile_options(-Werror=inconsistent-missing-override)
add_compile_options(-Werror=comment)
add_compile_options(-pedantic-errors)

# whitelist errors here
add_compile_options(-Weverything)
add_compile_options(-Wno-gnu-zero-variadic-macro-arguments)
add_compile_options(-Wno-c++98-compat-pedantic)
add_compile_options(-Wno-weak-vtables)
add_compile_options(-Wno-global-constructors)
add_compile_options(-Wno-exit-time-destructors)
add_compile_options(-Wno-missing-prototypes)
add_compile_options(-Wno-missing-noreturn)
add_compile_options(-Wno-covered-switch-default)
add_compile_options(-Wno-undef)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.1.0)
        add_compile_options(-Wno-zero-as-null-pointer-constant)
    endif()
endif()

# should remove these
add_compile_options(-Wno-float-conversion)
add_compile_options(-Wno-padded)
add_compile_options(-Wno-conversion)
add_compile_options(-Wno-undefined-func-template)
