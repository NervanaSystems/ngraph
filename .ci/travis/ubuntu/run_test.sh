#!/bin/bash
set -e

echo "TASK:" ${TASK}

if [ ${TASK} == "cpp_test" ]; then
    docker run -w '/root/ngraph/build' test_ngraph make check
fi

if [ ${TASK} == "python2_test" ]; then
    docker run -w '/root/ngraph/python' test_ngraph tox -e py27
fi

if [ ${TASK} == "python3_test" ]; then
    docker run -w '/root/ngraph/python' test_ngraph tox -e py3
fi
