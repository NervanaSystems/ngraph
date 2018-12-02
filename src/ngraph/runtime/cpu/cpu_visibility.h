//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

// https://gcc.gnu.org/wiki/Visibility
// Generic helper definitions for shared library support
#if defined _WIN32 || defined __CYGWIN__
#define CPU_HELPER_DLL_IMPORT __declspec(dllimport)
#define CPU_HELPER_DLL_EXPORT __declspec(dllexport)
#define CPU_HELPER_DLL_LOCAL
#else
#if __GNUC__ >= 4
#define CPU_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define CPU_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define CPU_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define CPU_HELPER_DLL_IMPORT
#define CPU_HELPER_DLL_EXPORT
#define CPU_HELPER_DLL_LOCAL
#endif
#endif

// Now we use the generic helper definitions above to define CPU_API and CPU_LOCAL.
// CPU_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)
// CPU_LOCAL is used for non-api symbols.

// #ifdef CPU_DLL         // defined if INTERPRETER is compiled as a DLL
#ifdef CPU_DLL_EXPORTS // defined if we are building the INTERPRETER DLL (instead of using it)
#define CPU_API CPU_HELPER_DLL_EXPORT
#else
#define CPU_API CPU_HELPER_DLL_IMPORT
#endif // CPU_DLL_EXPORTS
#define CPU_LOCAL CPU_HELPER_DLL_LOCAL
// #else // CPU_DLL is not defined: this means INTERPRETER is a static lib.
// #define CPU_API
// #define CPU_LOCAL
// #endif // CPU_DLL
