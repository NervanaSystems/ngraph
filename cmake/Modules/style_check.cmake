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

function(STYLE_CHECK_FILE PATH)
    # message(STATUS "*******xx******** ${PATH}")
endfunction()

set(DIRECTORIES_OF_INTEREST
    src
    test
)

find_program(CLANG_FORMAT clang-format PATHS ENV PATH)
message(STATUS "clang format search ${CLANG_FORMAT}")
if (CLANG_FORMAT)

endif()
message(STATUS "NGRAPH_SOURCE_DIR ${NGRAPH_SOURCE_DIR}")
foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST})
    set(DIR "${NGRAPH_SOURCE_DIR}/${DIRECTORY}/*.?pp")
    file(GLOB_RECURSE XPP_FILES ${DIR})
    foreach(FILE ${XPP_FILES})
        style_check_file(${FILE})
    endforeach(FILE)
endforeach(DIRECTORY)
