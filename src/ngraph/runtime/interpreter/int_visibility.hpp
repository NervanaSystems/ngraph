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
#define INTERPRETER_HELPER_DLL_IMPORT __declspec(dllimport)
#define INTERPRETER_HELPER_DLL_EXPORT __declspec(dllexport)
#define INTERPRETER_HELPER_DLL_LOCAL
#else
#if __GNUC__ >= 4
#define INTERPRETER_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define INTERPRETER_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define INTERPRETER_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define INTERPRETER_HELPER_DLL_IMPORT
#define INTERPRETER_HELPER_DLL_EXPORT
#define INTERPRETER_HELPER_DLL_LOCAL
#endif
#endif

// Now we use the generic helper definitions above to define INTERPRETER_API and INTERPRETER_LOCAL.
// INTERPRETER_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)
// INTERPRETER_LOCAL is used for non-api symbols.

// #ifdef INTERPRETER_DLL         // defined if INTERPRETER is compiled as a DLL
#ifdef INTERPRETER_DLL_EXPORTS // defined if we are building the INTERPRETER DLL (instead of using
                               // it)
#define INTERPRETER_API INTERPRETER_HELPER_DLL_EXPORT
#else
#define INTERPRETER_API INTERPRETER_HELPER_DLL_IMPORT
#endif // INTERPRETER_DLL_EXPORTS
#define INTERPRETER_LOCAL INTERPRETER_HELPER_DLL_LOCAL
// #else // INTERPRETER_DLL is not defined: this means INTERPRETER is a static lib.
// #define INTERPRETER_API
// #define INTERPRETER_LOCAL
// #endif // INTERPRETER_DLL
