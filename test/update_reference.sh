#!/bin/bash
declare THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python ${THIS_SCRIPT_DIR}/ref_generators/generate_convolution_ref.py ${THIS_SCRIPT_DIR}/convolution_test.in.cpp
#${THIS_SCRIPT_DIR}/../maint/apply-code-format.sh
