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

#pragma once

#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"
#include "ngraph/runtime/interpreter/int_backend_visibility.hpp"
#include "ngraph/runtime/nop/nop_backend_visibility.hpp"
#include "ngraph/runtime/plaidml/plaidml_backend_visibility.hpp"

extern "C" CPU_BACKEND_API void ngraph_register_cpu_backend();
extern "C" INTERPRETER_BACKEND_API void ngraph_register_interpreter_backend();
extern "C" PLAIDML_BACKEND_API void ngraph_register_plaidml_backend();
extern "C" NOP_BACKEND_API void ngraph_register_nop_backend();
