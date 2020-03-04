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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=inconsistent-missing-override")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=comment")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# whitelist errors here
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gnu-zero-variadic-macro-arguments")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-noreturn")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undef")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.1.0)
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-zero-as-null-pointer-constant")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c99-extensions")
endif()

# should remove these
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-func-template")
