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
#define NGRAPH_HELPER_DLL_IMPORT __declspec(dllimport)
#define NGRAPH_HELPER_DLL_EXPORT __declspec(dllexport)
#define NGRAPH_HELPER_DLL_LOCAL
#elif defined NGRAPH_LINUX_VISIBILITY_ENABLE && __GNUC__ >= 4
#define NGRAPH_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define NGRAPH_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#define NGRAPH_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
#else
#define NGRAPH_HELPER_DLL_IMPORT
#define NGRAPH_HELPER_DLL_EXPORT
#define NGRAPH_HELPER_DLL_LOCAL
#endif

// Now we use the generic helper definitions above to define NGRAPH_API and NGRAPH_LOCAL.
// NGRAPH_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)
// NGRAPH_LOCAL is used for non-api symbols.

#ifdef NGRAPH_DLL_EXPORTS // defined if we are building the NGRAPH DLL (instead of using it)
#define NGRAPH_API NGRAPH_HELPER_DLL_EXPORT
#else
#define NGRAPH_API NGRAPH_HELPER_DLL_IMPORT
#endif // NGRAPH_DLL_EXPORTS
#define NGRAPH_LOCAL NGRAPH_HELPER_DLL_LOCAL
