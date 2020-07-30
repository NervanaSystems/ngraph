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

if(TARGET pybind11::pybind11)
    return()
endif()

include(FetchContent)

message(STATUS "Fetching pybind11")

set(PYBIND11_GIT_TAG v2.5.0)
set(PYBIND11_ARCHIVE_HASH 8c78b412bed8b47210e74824d0bd8892f00aa023)
set(PYBIND11_ARCHIVE_URL https://github.com/pybind/pybind11/archive/${PYBIND11_GIT_TAG}.zip)

FetchContent_Declare(
    ext_pybind11
    URL            ${PYBIND11_ARCHIVE_URL}
    URL_HASH       SHA1=${PYBIND11_ARCHIVE_HASH}
)

FetchContent_GetProperties(ext_pybind11)
if(NOT ext_pybind11_POPULATED)
    FetchContent_Populate(ext_pybind11)
    add_subdirectory(${ext_pybind11_SOURCE_DIR} ${ext_pybind11_BINARY_DIR})
endif()
