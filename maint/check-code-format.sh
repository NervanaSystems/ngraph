#!/bin/bash
set -e
set -u

# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: The results of `clang-format` depend _both_ of the following factors:
# - The `.clang-format` file, and
# - The particular version of the `clang-format` program being used.
#
# For this reason, this script specifies the exact version of clang-format to be used.

declare CLANG_FORMAT_BASENAME="clang-format-3.9"

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    echo "Unable to find program ${CLANG_FORMAT_BASENAME}" >&2
    exit 1
fi

declare -a FAILED_FILES=()
declare NUM_FILES_CHECKED=0

pushd "${THIS_SCRIPT_DIR}/.."

declare ROOT_SUBDIR
for ROOT_SUBDIR in src test; do
    if ! [[ -d "${ROOT_SUBDIR}" ]]; then
	echo "In directory '$(pwd)', no subdirectory named '${ROOT_SUBDIR}' was found." >&2
	exit 1
    fi

    echo "About to format C/C++ code in directory tree '$(pwd)/${ROOT_SUBDIR}' ..."
    declare SRC_FILE
    for SRC_FILE in $(find "${ROOT_SUBDIR}" -type f -and \( -name '*.cpp' -or -name '*.hpp' \) ); do
        if "${CLANG_FORMAT_PROG}" -style=file -output-replacements-xml "${SRC_FILE}" | grep -c "<replacement " >/dev/null; then
            FAILED_FILES+=( "${SRC_FILE}" )
        fi
        NUM_FILES_CHECKED=$((NUM_FILES_CHECKED+1))
    done
done

popd

if [[ ${#FAILED_FILES[@]} -eq 0 ]]; then
    echo "All ${NUM_FILES_CHECKED}  C/C++ files pass the code-format check."
    exit 0
else
    echo "${#FAILED_FILES[@]} of ${NUM_FILES_CHECKED} source files failed the code-format check:"
    declare FAILED_SRC_FILE
    for FAILED_SRC_FILE in ${FAILED_FILES[@]}; do
        echo "    ${FAILED_SRC_FILE}"
    done
    exit 1
fi

