#!/bin/bash
set -e
set -u

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

if [[ $# -lt 1 ]]
then
    echo "Minimal arguments: apply-code-format.sh [path] ..."
    echo "e.g. apply-code-format.sh src test doc/examples"
    echo "     This will apply format to directories src, tests, and doc/examples"
    exit 1
fi

# NOTE: The results of `clang-format` depend _both_ of the following factors:
# - The `.clang-format` file, and
# - The particular version of the `clang-format` program being used.
#
# For this reason, this script specifies the exact version of clang-format to be used.

declare REQUIRED_CLANG_FORMAT_VERSION=3.9
declare CLANG_FORMAT_BASENAME="clang-format-"${REQUIRED_CLANG_FORMAT_VERSION}

declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${THIS_SCRIPT_DIR}/bash_lib.sh"
source "${THIS_SCRIPT_DIR}/clang_format_lib.sh"

declare CLANG_FORMAT_PROG
if ! CLANG_FORMAT_PROG="$(which "${CLANG_FORMAT_BASENAME}")"; then
    bash_lib_die "Unable to find program ${CLANG_FORMAT_BASENAME}" >&2
fi

clang_format_lib_verify_version "${CLANG_FORMAT_PROG}" "${REQUIRED_CLANG_FORMAT_VERSION}"
echo "Verified that '${CLANG_FORMAT_PROG}' has version '${REQUIRED_CLANG_FORMAT_VERSION}'"

declare DIR
for DIR in "$@"; do
    if ! [[ -d "${DIR}" ]]; then
        echo "No directory named '${DIR}' was found."
        exit 1
    else
        echo "Formatting C/C++ code in directory tree '${DIR}'"

        # Note that we restrict to "-type f" to exclude symlinks. Emacs sometimes
        # creates dangling symlinks with .cpp/.hpp suffixes as a sort of locking
        # mechanism, and this confuses clang-format.
        find "${DIR}" -type f -and \( -name '*.cpp' -or -name '*.hpp' \) | xargs "${CLANG_FORMAT_PROG}" -i -style=file
    fi
done
