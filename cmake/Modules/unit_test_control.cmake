# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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

function(UNIT_TEST_CONTROL)
    set(options)
    set(oneValueArgs BACKEND MANIFEST)
    set(multiValueArgs)
    cmake_parse_arguments(UNIT_TEST_CONTROL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if (UNIT_TEST_CONTROL_MANIFEST)
        get_filename_component(UNIT_TEST_CONTROL_MANIFEST ${UNIT_TEST_CONTROL_MANIFEST} ABSOLUTE)
        set(CONFIG_STRING "${UNIT_TEST_CONTROL_BACKEND}@${UNIT_TEST_CONTROL_MANIFEST}")
    else()
        set(CONFIG_STRING "${UNIT_TEST_CONTROL_BACKEND}@")
    endif()
    set(UNIT_TEST_CONFIG_LIST "${UNIT_TEST_CONFIG_LIST};${CONFIG_STRING}" CACHE INTERNAL "")
endfunction()
