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

// #include <fstream>
#include <memory>
#include <string>
// #include <tuple>
// #include <typeindex>
// #include <typeinfo>
#include <unordered_map>

#include "ngraph/runtime/gpu/external_function.hpp"
#include "ngraph/runtime/gpu/call_frame.hpp"
#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph::runtime::gpu;
using namespace ngraph;

ngraph::runtime::gpu::ExternalFunction::ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                         bool release_function)
    : runtime::ExternalFunction(function, release_function)
  , m_function(function)
{
}

void runtime::gpu::ExternalFunction::compile()
{
  // if (m_is_compiled)
  //   {
  //     return;
  //   }

  // string function_name = m_function->get_name();
  // string dump_filename = file_util::path_join(s_output_dir, function_name + "_ops.txt");

  // pass::Manager pass_manager;
  // pass_manager.register_pass<pass::TopologicalSort>();
  // // For now, just make everyone row-major.
  // pass_manager.register_pass<pass::AssignLayout<DenseTensorViewLayout>>();
  // pass_manager.register_pass<pass::Liveness>();
  // pass_manager.run_passes(m_function);

  // m_is_compiled = true;
  // if (m_release_function)
  //   {
  //     release_function();
  //   }
}

shared_ptr<runtime::CallFrame> runtime::gpu::ExternalFunction::make_call_frame()
{
  if (!m_is_compiled)
    {
      compile();
    }

  return make_shared<runtime::gpu::GPUCallFrame>(shared_from_this(), m_function);
}

// GPUExternalFunction::~GPUExternalFunction(){}
