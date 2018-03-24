/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/external_function.hpp"

using namespace ngraph;
using namespace std;

std::shared_ptr<ngraph::runtime::CallFrame> runtime::cpu::CPU_Backend::make_call_frame(
    const std::shared_ptr<runtime::ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

std::shared_ptr<ngraph::runtime::TensorView>
    runtime::cpu::CPU_Backend::make_primary_tensor_view(const ngraph::element::Type& element_type,
                                                        const Shape& shape)
{
    auto rc = make_shared<runtime::cpu::CPUTensorView>(element_type, shape);
    return dynamic_pointer_cast<runtime::TensorView>(rc);
}

std::shared_ptr<ngraph::runtime::TensorView>
    runtime::cpu::CPU_Backend::create_tensor(const ngraph::element::Type& element_type,
                                             const Shape& shape)
{
    auto rc = make_shared<runtime::cpu::CPUTensorView>(element_type, shape);
    return dynamic_pointer_cast<runtime::TensorView>(rc);
}

bool runtime::cpu::CPU_Backend::compile(const ngraph::Function& func)
{
    m_function = clone_function(func);
    if (m_external_function)
    {
        throw runtime_error("Backend can only compile a single function");
    }
    m_external_function = make_shared<cpu::CPU_ExternalFunction>(m_function);
    auto cf = m_external_function->make_call_frame();
    m_call_frame = dynamic_pointer_cast<cpu::CPU_CallFrame>(cf);
    return true;
}

bool runtime::cpu::CPU_Backend::is_callable() const
{
    throw runtime_error("Unimplemented");
}

bool runtime::cpu::CPU_Backend::call(
    const std::vector<std::shared_ptr<runtime::TensorView>>& outputs,
    const std::vector<std::shared_ptr<runtime::TensorView>>& inputs)
{
    bool rc = false;
    if (m_call_frame)
    {
        m_call_frame->call(outputs, inputs);
        rc = true;
    }
    return rc;
}

vector<size_t> runtime::cpu::CPU_Backend::get_subdevices() const
{
    vector<size_t> rc;
    return rc;
}
