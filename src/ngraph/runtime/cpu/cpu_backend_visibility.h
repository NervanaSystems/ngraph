//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#define CPU_BACKEND_HELPER_DLL_IMPORT __declspec(dllimport)
#define CPU_BACKEND_HELPER_DLL_EXPORT __declspec(dllexport)
#define CPU_BACKEND_HELPER_DLL_LOCAL
#else
#if __GNUC__ >= 4
#define CPU_BACKEND_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define CPU_BACKEND_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define CPU_BACKEND_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define CPU_BACKEND_HELPER_DLL_IMPORT
#define CPU_BACKEND_HELPER_DLL_EXPORT
#define CPU_BACKEND_HELPER_DLL_LOCAL
#endif
#endif

// Now we use the generic helper definitions above to define CPU_BACKEND_API and CPU_BACKEND_LOCAL.
// CPU_BACKEND_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)
// CPU_BACKEND_LOCAL is used for non-api symbols.

// #ifdef CPU_BACKEND_DLL         // defined if CPU_BACKEND is compiled as a DLL
#ifdef CPU_BACKEND_DLL_EXPORTS // defined if we are building the CPU_BACKEND DLL (instead of using it)
#define CPU_BACKEND_API CPU_BACKEND_HELPER_DLL_EXPORT
#else
#define CPU_BACKEND_API CPU_BACKEND_HELPER_DLL_IMPORT
#endif // CPU_BACKEND_DLL_EXPORTS
#define CPU_BACKEND_LOCAL CPU_BACKEND_HELPER_DLL_LOCAL
// #else // CPU_BACKEND_DLL is not defined: this means CPU_BACKEND is a static lib.
// #define CPU_BACKEND_API
// #define CPU_BACKEND_LOCAL
// #endif // CPU_BACKEND_DLL
