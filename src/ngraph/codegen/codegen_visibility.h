//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#define CODEGEN_HELPER_DLL_IMPORT __declspec(dllimport)
#define CODEGEN_HELPER_DLL_EXPORT __declspec(dllexport)
#define CODEGEN_HELPER_DLL_LOCAL
#else
#if __GNUC__ >= 4
#define CODEGEN_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define CODEGEN_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define CODEGEN_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define CODEGEN_HELPER_DLL_IMPORT
#define CODEGEN_HELPER_DLL_EXPORT
#define CODEGEN_HELPER_DLL_LOCAL
#endif
#endif

// Now we use the generic helper definitions above to define CODEGEN_API and CODEGEN_LOCAL.
// CODEGEN_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)
// CODEGEN_LOCAL is used for non-api symbols.

// #ifdef CODEGEN_DLL         // defined if CODEGEN is compiled as a DLL
#ifdef CODEGEN_DLL_EXPORTS // defined if we are building the CODEGEN DLL (instead of using it)
#define CODEGEN_API CODEGEN_HELPER_DLL_EXPORT
#else
#define CODEGEN_API CODEGEN_HELPER_DLL_IMPORT
#endif // CODEGEN_DLL_EXPORTS
#define CODEGEN_LOCAL CODEGEN_HELPER_DLL_LOCAL
// #else // CODEGEN_DLL is not defined: this means CODEGEN is a static lib.
// #define CODEGEN_API
// #define CODEGEN_LOCAL
// #endif // CODEGEN_DLL
