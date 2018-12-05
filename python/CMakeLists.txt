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

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    message(FATAL_ERROR "Please call from nGraph !")
endif()

if (WIN32)
    message(WARNING "Python API is not supported on Windows!")
    return()
endif()

if (NOT NGRAPH_CPU_ENABLE OR NOT NGRAPH_DEX_ONLY)
    message(WARNING "Python API requires NGRAPH_CPU_ENABLE and NGRAPH_DEX_ONLY!")
    return()
endif()

include(ExternalProject)

set(PYBIND11_SOURCE_DIR "${CMAKE_BINARY_DIR}/pybind11")

ExternalProject_Add(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
        GIT_TAG "v2.2.4"
        SOURCE_DIR ${PYBIND11_SOURCE_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

set(NGRAPH_DEPS_INCLUDE ${CMAKE_SOURCE_DIR}/src)
set(NGRAPH_DEPS_LIB     ${NGRAPH_BUILD_DIR})

set(SETUP_PY_IN "${CMAKE_SOURCE_DIR}/python/setup.py.in")
set(SETUP_PY    "${CMAKE_BINARY_DIR}/setup.py")
configure_file(${SETUP_PY_IN} ${SETUP_PY})

set(TOX_INI_IN "${CMAKE_SOURCE_DIR}/python/tox.ini.in")
set(TOX_INI    "${CMAKE_BINARY_DIR}/tox.ini")
configure_file(${TOX_INI_IN} ${TOX_INI})

unset(PYTHON_EXECUTABLE)
unset(PYTHONINTERP_FOUND)
find_package(PythonInterp 2)
if (PYTHON_EXECUTABLE)
    set(PYTHON2_EXECUTABLE ${PYTHON_EXECUTABLE})
    find_program(VIRTUALENV virtualenv)
    if (NOT VIRTUALENV)
        message(FATAL_ERROR "Please install virtualenv!")
    endif()

    message(STATUS "Python2 executable:" ${PYTHON_EXECUTABLE})

    add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/wheel2.timestamp
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMAND ${VIRTUALENV} -p ${PYTHON2_EXECUTABLE} .venv2
            COMMAND ./.venv2/bin/python -m pip install -U pip setuptools wheel
            COMMAND ./.venv2/bin/python setup.py bdist_wheel
            COMMAND ${CMAKE_COMMAND} -E touch wheel3.timestamp
            DEPENDS pybind11 ngraph cpu_backend interpreter_backend
    )
    list(APPEND WHEEL_DEPS ${CMAKE_BINARY_DIR}/wheel2.timestamp)
endif (PYTHON_EXECUTABLE)

if (NOT NGRAPH_MANYLINUX_ENABLE)
    # Workaround: find_package(PythonInterp 3) does not work.
    find_program(PYTHON3_EXECUTABLE python3)
    if (PYTHON3_EXECUTABLE)
        message(STATUS "Python3 executable:" ${PYTHON3_EXECUTABLE})

        add_custom_command(
                OUTPUT ${CMAKE_BINARY_DIR}/wheel3.timestamp
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                COMMAND ${PYTHON3_EXECUTABLE} -m venv .venv3
                COMMAND ./.venv3/bin/python -m pip install -U pip setuptools wheel
                COMMAND ./.venv3/bin/python setup.py bdist_wheel
                COMMAND ${CMAKE_COMMAND} -E touch wheel3.timestamp
                DEPENDS pybind11 ngraph cpu_backend interpreter_backend ${WHEEL_DEPS}
        )
        list(APPEND WHEEL_DEPS ${CMAKE_BINARY_DIR}/wheel3.timestamp)
    endif (PYTHON3_EXECUTABLE)
endif()

add_custom_target(python_wheel DEPENDS ${WHEEL_DEPS})

