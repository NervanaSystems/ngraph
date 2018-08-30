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

#pragma once

#include <memory>

#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class TensorViewWrapper;
        }
    }
}

class ngraph::runtime::cpu::TensorViewWrapper
{
public:
    TensorViewWrapper(const std::shared_ptr<descriptor::TensorView>&,
                      const std::string& alias = "");

    size_t get_size() const;
    const Shape& get_shape() const;
    const Strides& get_strides() const;
    const element::Type& get_element_type() const;
    const std::string& get_name() const;
    const std::string& get_type() const;
    const std::shared_ptr<descriptor::TensorView> get_tensor_view() const;

private:
    std::shared_ptr<descriptor::TensorView> m_tensor_view;
    std::string m_alias;
};
