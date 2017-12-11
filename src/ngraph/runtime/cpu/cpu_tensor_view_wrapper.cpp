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

#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::TensorViewWrapper::TensorViewWrapper(const shared_ptr<descriptor::TensorView>& tv)
    : m_tensor_view(tv)
{
}

size_t runtime::cpu::TensorViewWrapper::get_size() const
{
    return m_tensor_view->get_tensor_view_layout()->get_size();
}

const vector<size_t>& runtime::cpu::TensorViewWrapper::get_shape() const
{
    return m_tensor_view->get_tensor_view_layout()->get_shape();
}

const vector<size_t>& runtime::cpu::TensorViewWrapper::get_strides() const
{
    return m_tensor_view->get_tensor_view_layout()->get_strides();
}

const element::Type& runtime::cpu::TensorViewWrapper::get_element_type() const
{
    return m_tensor_view->get_tensor_view_layout()->get_element_type();
}

const std::string& runtime::cpu::TensorViewWrapper::get_name() const
{
    return m_tensor_view->get_tensor().get_name();
}

const std::string& runtime::cpu::TensorViewWrapper::get_type() const
{
    return get_element_type().c_type_string();
}

bool runtime::cpu::TensorViewWrapper::is_output() const
{
    return m_tensor_view->get_tensor().is_output();
}
