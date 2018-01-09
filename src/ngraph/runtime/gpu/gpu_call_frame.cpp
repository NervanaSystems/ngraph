// ----------------------------------------------------------------------------
// copyright 2017 nervana systems inc.
// licensed under the apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
//      http://www.apache.org/licenses/license-2.0
//
// unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// see the license for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <fstream>


#include "ngraph/runtime/gpu/gpu_call_frame.hpp"

using namespace std;
using namespace ngraph::runtime::gpu;

GPU_CallFrame::GPU_CallFrame(shared_ptr<GPU_ExternalFunction> external_function,
                             shared_ptr<Function> func)
    : m_external_function(external_function)
    , m_function(func)
{
}
