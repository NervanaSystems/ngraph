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

#include <cstring>
#include <memory>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/runtime/chrome_trace.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/interpreter/int_tensor.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

static const size_t alignment = 64;

runtime::interpreter::INTTensor::INTTensor(const ngraph::element::Type& element_type,
                                           const Shape& shape,
                                           const string& name)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, shape, name))
    , m_tensor_value(element_type, shape)
{
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));
}

runtime::interpreter::INTTensor::INTTensor(const ngraph::element::Type& element_type,
                                           const Shape& shape)
    : INTTensor(element_type, shape, "")
{
}

runtime::interpreter::INTTensor::~INTTensor()
{
}

void runtime::interpreter::INTTensor::write(const void* source, size_t n)
{
    if (n > shape_size(m_tensor_value.shape()) * m_tensor_value.element_type().size())
    {
        throw out_of_range("write access past end of tensor");
    }
    memcpy(m_tensor_value.raw_buffer(), source, n);
}

void runtime::interpreter::INTTensor::read(void* target, size_t n) const
{
    if (n > shape_size(m_tensor_value.shape()) * m_tensor_value.element_type().size())
    {
        throw out_of_range("read access past end of tensor");
    }
    std::cout << m_tensor_value << std::endl;
    memcpy(target, m_tensor_value.raw_buffer(), n);
}
