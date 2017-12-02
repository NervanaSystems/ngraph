// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/runtime/gpu/external_function.hpp"
#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph::runtime::gpu;

GPUExternalFunction::GPUExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                         bool release_function)
    : ngraph::runtime::ExternalFunction(function, release_function)
  , m_compiled_function(nullptr)
{
}
