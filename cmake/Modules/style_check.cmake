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

set(CLANG_FORMAT_FILENAME clang-format-3.9)
find_program(CLANG_FORMAT ${CLANG_FORMAT_FILENAME} PATHS ENV PATH)

function(STYLE_CHECK_FILE PATH)
    execute_process(COMMAND ${CLANG_FORMAT} -style=file -output-replacements-xml ${PATH}
        OUTPUT_VARIABLE STYLE_CHECK_RESULT)
        list(LENGTH STYLE_CHECK_RESULT RESULT_LENGTH)
        # message(STATUS "${PATH} ${RESULT_LENGTH}")
        if (RESULT_LENGTH GREATER 1)
            message(STATUS "style error ${PATH}")
        endif()
endfunction()

set(DIRECTORIES_OF_INTEREST
    src
    doc
    test
    python/pyngraph
)

if (CLANG_FORMAT)
    foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST})
        set(DIR "${NGRAPH_SOURCE_DIR}/${DIRECTORY}/*.?pp")
        file(GLOB_RECURSE XPP_FILES ${DIR})
        foreach(FILE ${XPP_FILES})
            style_check_file(${FILE})
        endforeach(FILE)
    endforeach(DIRECTORY)
else()
    message(STATUS "${CLANG_FORMAT_FILENAME} not found, style not available")
endif()
