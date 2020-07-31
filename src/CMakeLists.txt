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

#-----------------------------------------------------------------------------------------------
# Compile Flags for nGraph...
#-----------------------------------------------------------------------------------------------

add_compile_options(${NGRAPH_COMMON_FLAGS})

if (WIN32)
    string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Compiler-specific logic...
if (WIN32)
    add_compile_options(/EHsc /MP)

    if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        add_compile_options(/bigobj)
        # C4251 needs to have dll-interface to be used by clients of class
        add_compile_options(/wd4251)
        # C4275 non dll-interface class used as base for dll-interface class
        add_compile_options(/wd4275)
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
    set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_C_VISIBILITY_PRESET default)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(AppleClang)?Clang$" AND NOT NGRAPH_BUILD_DIR)
    message( STATUS "Setting clang flags...")
    include( ${PROJECT_SOURCE_DIR}/cmake/clang_flags.cmake )
endif()

if (NOT WIN32 AND ${NGRAPH_WARNINGS_AS_ERRORS})
    add_compile_options(-Werror)
endif()

if (NGRAPH_CODE_COVERAGE_ENABLE)
    add_compile_options(--coverage)
endif()

# Enable build target CPU features
if(DEFINED NGRAPH_TARGET_ARCH)
    add_compile_options(-march=${NGRAPH_TARGET_ARCH})
elseif(NOT WIN32)
    add_compile_options(-march=core2 -msse4.1)
endif()

if (DEFINED NGRAPH_TUNE_ARCH)
    add_compile_options(-mtune=${NGRAPH_TUNE_ARCH})
endif()

if (NGRAPH_USE_GOLD)
    add_compile_options(-fuse-ld=gold)
endif()
if(WIN32)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

if (NGRAPH_MLIR_ENABLE OR NGRAPH_CPU_MLIR_ENABLE)
    set(NGRAPH_MLIR_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/contrib/mlir)
endif()

if (NOT DEFINED NGRAPH_TBB_ENABLE)
    set(NGRAPH_TBB_ENABLE ${NGRAPH_CPU_ENABLE})
endif()

if(NGRAPH_ADDRESS_SANITIZER_ENABLE)
    if (APPLE)
        set(ADDRESS_SANITIZER_FLAGS "-fsanitize=address")
    else()
        set(ADDRESS_SANITIZER_FLAGS "-fsanitize=address -fsanitize=leak")
    endif()
    add_compile_options(${ADDRESS_SANITIZER_FLAGS})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
endif()

if(NGRAPH_THREAD_SANITIZER_ENABLE)
    add_compile_options(-fsanitize=thread)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
endif()

if(NGRAPH_UB_SANITIZER_ENABLE)
    add_compile_options(-fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
endif()

if(NGRAPH_ADDRESS_SANITIZER_ENABLE OR NGRAPH_THREAD_SANITIZER_ENABLE OR NGRAPH_UB_SANITIZER_ENABLE)
    if (LINUX AND NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "^Clang$")
        message(FATAL_ERROR "Sanitizers only work with the clang compiler")
    endif()
    add_compile_options(-g -fno-omit-frame-pointer)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
endif()

#-----------------------------------------------------------------------------------------------
# enable or disable output from NGRAPH_DEBUG statements
#-----------------------------------------------------------------------------------------------
if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
endif()

#-----------------------------------------------------------------------------------------------
# enable or disable deprecation warnings for old APIs
#-----------------------------------------------------------------------------------------------
if(NGRAPH_DEPRECATED_ENABLE)
    add_definitions(-DNGRAPH_DEPRECATED_ENABLE)
endif()

add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")

#-----------------------------------------------------------------------------------------------
# Print Options
#-----------------------------------------------------------------------------------------------
get_directory_property(NGRAPH_COMPILE_OPTIONS COMPILE_OPTIONS)
message(STATUS "Compile Flags: ${NGRAPH_COMPILE_OPTIONS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_CXX_FLAGS_MINSIZEREL ${CMAKE_CXX_FLAGS_MINSIZEREL}")

add_definitions(-DIN_NGRAPH_LIBRARY)

include_directories(ngraph)

add_subdirectory(resource)

# This must be added before any backend that uses MLIR
add_subdirectory(contrib)

add_subdirectory(ngraph)

add_subdirectory(runtime)

if (NGRAPH_TOOLS_ENABLE)
    add_subdirectory(tools)
    message(STATUS "tools enabled")
else()
    message(STATUS "tools disabled")
endif()
