#!/usr/bin/env bash

###############################################################################
# 
# This script generates stub files for automatically generated documentation
# of the nGraph Python API.
# 
###############################################################################

# paths relative to this file location
CURRENT_DIR="$(pwd)"
NGRAPH_REPO="${CURRENT_DIR}/../.."
DOC_DIR=${NGRAPH_REPO}/doc
TMP_DIR=/tmp/sphinx_auto_py_doc
EXCLUDE_DIRS="${NGRAPH_REPO}/python/ngraph/impl*
              ${NGRAPH_REPO}/python/ngraph/utils*"

pushd ${NGRAPH_REPO}/python
PYTHONPATH="$(pwd)" sphinx-autogen -t ${DOC_DIR}/sphinx/source/_templates/ -o ${TMP_DIR} \
                                      ${DOC_DIR}/sphinx/source/python_api/structure.rst
popd
sphinx-apidoc -f -M -d 1 -T -o ${TMP_DIR} ${CURRENT_DIR} ${EXCLUDE_DIRS}

cp ${TMP_DIR}/* ${DOC_DIR}/sphinx/source/python_api/_autosummary/

rm -rf ${TMP_DIR}
