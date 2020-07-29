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

if(TARGET nlohmann_json::nlohmann_json)
    return()
endif()

include(FetchContent)

message(STATUS "Fetching nlohmann json")

# Hedley annotations introduced in v3.7.0 causes build failure on MSVC 2017 + ICC 18
if(WIN32 AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
    SET(JSON_GIT_LABEL v3.6.1)
else()
    SET(JSON_GIT_LABEL v3.8.0)
endif()

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG        ${JSON_GIT_LABEL}
    GIT_SHALLOW    1)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
