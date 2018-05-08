#!/usr/bin/env bash

###############################################################################
# 
# This script generates stub files for automatically generated documentation
# of the nGraph Python API.
# 
###############################################################################

# paths relative to this file location
NGRAPH_REPO=../..
DOC_DIR=${NGRAPH_REPO}/doc
TMP_DIR=/tmp/sphinx_auto_py_doc
EXCLUDE_DIRS="${NGRAPH_REPO}/python/ngraph/impl*
              ${NGRAPH_REPO}/python/ngraph/utils*"
CURRENT_DIR=.

cd ${NGRAPH_REPO}/python/ngraph
PYTHONPATH=. sphinx-autogen -t ${DOC_DIR}/sphinx/source/_templates/ -o ${TMP_DIR} \
                             ${DOC_DIR}/sphinx/source/python_api/structure.rst
sphinx-apidoc -f -M -d 1 -T -o ${TMP_DIR} ${CURRENT_DIR} ${EXCLUDE_DIRS}

rm ${TMP_DIR}/ngraph.runtime.rst
cp ${TMP_DIR}/* ${DOC_DIR}/sphinx/source/python_api/_autosummary/

rm -rf ${TMP_DIR}
