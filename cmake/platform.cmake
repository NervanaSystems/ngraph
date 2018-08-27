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

if (EXISTS "/etc/lsb-release")
    file(STRINGS "/etc/lsb-release" __lsb_release_file_content)
    foreach(__line ${__lsb_release_file_content})
        string(STRIP "${__line}" __line)
        string(REGEX MATCH "^(.+)=(.*)$" __lsb_release_values_check ${__line})
        set(${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
    endforeach()
elseif (EXISTS "/etc/os-release")
    file(STRINGS "/etc/os-release" __os_release_file_content)
    foreach(__line ${__os_release_file_content})
        string(STRIP "${__line}" __line)
        string(REGEX MATCH "^(.+)=(.*)$" __os_release_values_check ${__line})
        string(REPLACE "\"" "" __variable_value ${CMAKE_MATCH_2})
        set(__${CMAKE_MATCH_1} ${__variable_value})
    endforeach()
    set(DISTRIB_ID ${__NAME})
    set(DISTRIB_CODENAME ${__VERSION_CODENAME})
    set(DISTRIB_RELEASE ${__VERSION_ID})
    set(DISTRIB_DESCRIPTION ${__PRETTY_NAME})
else()
    unset(DISTRIB_ID "?unknown?")
    unset(DISTRIB_CODENAME)
    unset(DISTRIB_RELEASE)
    unset(DISTRIB_DESCRIPTION)
endif()
