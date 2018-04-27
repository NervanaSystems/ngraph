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

function(NGRAPH_GET_CURRENT_HASH)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --verify HEAD
        RESULT_VARIABLE result
        OUTPUT_VARIABLE HASH)
    string(STRIP ${HASH} HASH)
    set(NGRAPH_CURRENT_HASH ${HASH} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_TAG_OF_CURRENT_HASH)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} ls-remote --tags
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE TAG_LIST)
    NGRAPH_GET_CURRENT_HASH()

    string(REGEX MATCH "${NGRAPH_CURRENT_HASH}\t[^\r\n]*" TAG ${TAG_LIST})
    set(FINAL_TAG ${TAG})
    if (NOT "${TAG}" STREQUAL "")
        string(REGEX REPLACE "${NGRAPH_CURRENT_HASH}\trefs/tags/(.*)" "\\1" FINAL_TAG ${TAG})
    endif()
    set(NGRAPH_CURRENT_RELEASE_TAG ${FINAL_TAG} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_MOST_RECENT_TAG)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        RESULT_VARIABLE RESULT
        OUTPUT_VARIABLE TAG)
    string(STRIP ${TAG} TAG)
    set(NGRAPH_MOST_RECENT_RELEASE_TAG ${TAG} PARENT_SCOPE)
endfunction()

function(NGRAPH_GET_VERSION_LABEL)
    NGRAPH_GET_TAG_OF_CURRENT_HASH()
    set(NGRAPH_VERSION_LABEL ${NGRAPH_CURRENT_RELEASE_TAG} PARENT_SCOPE)
    if ("${NGRAPH_CURRENT_RELEASE_TAG}" STREQUAL "")
        NGRAPH_GET_CURRENT_HASH()
        NGRAPH_GET_MOST_RECENT_TAG()
        string(SUBSTRING "${NGRAPH_CURRENT_HASH}" 0 7 HASH)
        set(NGRAPH_VERSION_LABEL "${NGRAPH_MOST_RECENT_RELEASE_TAG}+${HASH}" PARENT_SCOPE)
    endif()
endfunction()
