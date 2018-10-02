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

#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"

using namespace std;
using namespace ngraph;

runtime::gpu::GPUTensorWrapper::GPUTensorWrapper(const shared_ptr<descriptor::Tensor>& tv,
                                                 const string& alias)
    : m_tensor(tv)
    , m_alias(alias)
{
}

size_t runtime::gpu::GPUTensorWrapper::get_size() const
{
    return m_tensor->get_tensor_layout()->get_size();
}

const Shape& runtime::gpu::GPUTensorWrapper::get_shape() const
{
    return m_tensor->get_tensor_layout()->get_shape();
}

Strides runtime::gpu::GPUTensorWrapper::get_strides() const
{
    return m_tensor->get_tensor_layout()->get_strides();
}

const element::Type& runtime::gpu::GPUTensorWrapper::get_element_type() const
{
    return m_tensor->get_tensor_layout()->get_element_type();
}

const std::string& runtime::gpu::GPUTensorWrapper::get_name() const
{
    if (m_alias.empty())
    {
        return m_tensor->get_name();
    }
    else
    {
        return m_alias;
    }
}

const std::string& runtime::gpu::GPUTensorWrapper::get_type() const
{
    return get_element_type().c_type_string();
}
