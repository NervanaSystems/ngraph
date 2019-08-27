# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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
"""Op class generator for nGraph"""


from opgen.writer import ClassWriter

import sys
import argparse
import os

def main(args=sys.argv):
    parser = argparse.ArgumentParser('opgen')
    parser.add_argument('--defn', metavar='filename.json', type=argparse.FileType('r'),
                        nargs=1, action='store', required=True, help='Source class definition file')
    parser.add_argument('--outdir', metavar='/path/to/ngraph/src', type=str,
                        nargs=1, action='store', required=True, help='ngraph/src directory for output')
    flags = parser.parse_args()

    [json_file] = flags.defn
    [outdir_name] = flags.outdir

    os.makedirs(outdir_name, exist_ok=True)

    generator = ClassWriter(json_file)
    generator.gen_hpp_file(outdir_name)
    generator.gen_cpp_file(outdir_name)


if __name__ == "__main__":
    main()
