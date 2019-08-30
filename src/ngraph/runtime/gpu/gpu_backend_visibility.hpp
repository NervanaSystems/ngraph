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

#include "ngraph/visibility.hpp"

// Now we use the generic helper definitions above to define GPU_BACKEND_API
// GPU_BACKEND_API is used for the public API symbols. It either DLL imports or DLL exports
//    (or does nothing for static build)

#ifdef GPU_BACKEND_EXPORTS // defined if we are building the GPU_BACKEND
#define GPU_BACKEND_API NGRAPH_HELPER_DLL_EXPORT
#else
#define GPU_BACKEND_API NGRAPH_HELPER_DLL_IMPORT
#endif
