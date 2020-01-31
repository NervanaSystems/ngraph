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

if(NGRAPH_MANYLINUX_ENABLE)
    if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
        message(FATAL_ERROR "Please call from nGraph !")
    endif()

    if (WIN32)
        message(WARNING "Python API is not supported on Windows!")
        return()
    endif()
else()
    cmake_minimum_required (VERSION 3.1)

    project (pyngraph)
endif()

include(ExternalProject)

set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11")

ExternalProject_Add(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
        GIT_TAG "v2.2.4"
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pybind11"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

set(BUILD_SH_IN "${CMAKE_CURRENT_SOURCE_DIR}/configure_file/build_wheel.sh.in")
set(BUILD_SH    "${CMAKE_CURRENT_BINARY_DIR}/build_wheel.sh")
configure_file(${BUILD_SH_IN} ${BUILD_SH} @ONLY)

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dist/
        POST_BUILD
        WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${CMAKE_BUILD_TOOL} -C ../ DESTDIR=python/_install install && bash build_wheel.sh
)

add_custom_target(python_wheel DEPENDS ngraph ${CMAKE_CURRENT_BINARY_DIR}/dist/)
if (NGRAPH_CPU_ENABLE)
    add_dependencies(python_wheel libmkldnn)
endif()

if (NOT NGRAPH_MANYLINUX_ENABLE)
    return()
endif()

set(NGRAPH_DEPS_INCLUDE ${CMAKE_SOURCE_DIR}/src)
set(NGRAPH_DEPS_LIB     ${NGRAPH_BUILD_DIR})

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/configure_file/setup.in.py")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
configure_file(${SETUP_PY_IN} ${SETUP_PY})

set(TOX_INI_IN "${CMAKE_CURRENT_SOURCE_DIR}/configure_file/tox.in.ini")
set(TOX_INI    "${CMAKE_CURRENT_BINARY_DIR}/tox.ini")
configure_file(${TOX_INI_IN} ${TOX_INI})

list(APPEND WHEEL_DEP_LIBS pybind11)
list(APPEND WHEEL_DEP_LIBS ngraph)
list(APPEND WHEEL_DEP_LIBS interpreter_backend)
if(NGRAPH_CPU_ENABLE)
    list(APPEND WHEEL_DEP_LIBS cpu_backend)
endif()
if(NGRAPH_GPU_ENABLE)
    list(APPEND WHEEL_DEP_LIBS gpu_backend)
endif()
if(NGRAPH_INTELGPU_ENABLE)
    list(APPEND WHEEL_DEP_LIBS intelgpu_backend)
endif()
if(NGRAPH_PLAIDML_ENABLE)
    list(APPEND WHEEL_DEP_LIBS plaidml_backend)
endif()
if(NGRAPH_HYBRID_ENABLE)
    list(APPEND WHEEL_DEP_LIBS hybrid_backend)
endif()
if (NGRAPH_USE_LLVM_OMPLIB)
    if(NOT NGRAPH_CODEGEN_ENABLE)
        list(APPEND WHEEL_DEP_LIBS ext_omprt)
    endif()
endif()

if(LINUX)
    find_program(PATCHELF patchelf)
    if(PATCHELF)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/patchelf.timestamp
            WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${PATCHELF} --force-rpath --set-rpath \'$$ORIGIN\' ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLML_LIB}
            COMMAND ${PATCHELF} --set-soname ${MKLML_LIB}  ${NGRAPH_LIBRARY_OUTPUT_DIRECTORY}/${MKLML_LIB}
            COMMAND ${CMAKE_COMMAND} -E touch patchelf.timestamp
            DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
        )
        list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/patchelf.timestamp)
    endif()
endif()

if (NOT NGRAPH_MANYLINUX_ENABLE)
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
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel2.timestamp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${VIRTUALENV} -p ${PYTHON2_EXECUTABLE} .venv2
            COMMAND ./.venv2/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
            COMMAND ./.venv2/bin/python setup.py bdist_wheel
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
            COMMAND ${CMAKE_COMMAND} -E touch wheel2.timestamp
            DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
        )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel2.timestamp)
    endif (PYTHON_EXECUTABLE)

    # Workaround: find_package(PythonInterp 3) does not work.
    find_program(PYTHON3_EXECUTABLE python3)
    if (PYTHON3_EXECUTABLE)
        message(STATUS "Python3 executable:" ${PYTHON3_EXECUTABLE})

        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel3.timestamp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${PYTHON3_EXECUTABLE} -m venv .venv3
            COMMAND ./.venv3/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
            COMMAND ./.venv3/bin/python setup.py bdist_wheel
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
            COMMAND ${CMAKE_COMMAND} -E touch wheel3.timestamp
            DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
        )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel3.timestamp)
    endif (PYTHON3_EXECUTABLE)
else()
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel27mu.timestamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND /opt/python/cp27-cp27mu/bin/virtualenv -p /opt/python/cp27-cp27mu/bin/python2.7 .venv27mu
        COMMAND ./.venv27mu/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
        COMMAND ./.venv27mu/bin/python setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
        COMMAND ${CMAKE_COMMAND} -E touch wheel27mu.timestamp
        DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
    )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel27mu.timestamp)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel27m.timestamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND /opt/python/cp27-cp27mu/bin/virtualenv -p /opt/python/cp27-cp27m/bin/python2.7 .venv27m
        COMMAND ./.venv27m/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
        COMMAND ./.venv27m/bin/python setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
        COMMAND ${CMAKE_COMMAND} -E touch wheel27m.timestamp
        DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
    )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel27m.timestamp)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel34.timestamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND /opt/python/cp34-cp34m/bin/python3.4 -m venv .venv34
        COMMAND ./.venv34/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
        COMMAND ./.venv34/bin/python setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
        COMMAND ${CMAKE_COMMAND} -E touch wheel34.timestamp
        DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
    )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel34.timestamp)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel35.timestamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND /opt/python/cp35-cp35m/bin/python3.5 -m venv .venv35
        COMMAND ./.venv35/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
        COMMAND ./.venv35/bin/python setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
        COMMAND ${CMAKE_COMMAND} -E touch wheel35.timestamp
        DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
    )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel35.timestamp)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel36.timestamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND /opt/python/cp36-cp36m/bin/python3.6 -m venv .venv36
        COMMAND ./.venv36/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
        COMMAND ./.venv36/bin/python setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
        COMMAND ${CMAKE_COMMAND} -E touch wheel36.timestamp
        DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
    )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel36.timestamp)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/wheel37.timestamp
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND /opt/python/cp37-cp37m/bin/python3.7 -m venv .venv37
        COMMAND ./.venv37/bin/python -m pip install --no-cache-dir -U pip setuptools wheel
        COMMAND ./.venv37/bin/python setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/ngraph_core.egg-info
        COMMAND ${CMAKE_COMMAND} -E touch wheel37.timestamp
        DEPENDS ${WHEEL_DEP_LIBS} ${WHEEL_DEPS}
    )
    list(APPEND WHEEL_DEPS ${CMAKE_CURRENT_BINARY_DIR}/wheel37.timestamp)
endif()

add_custom_target(manylinux_wheel DEPENDS ${WHEEL_DEPS})

