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

#pragma once

#include <memory>

#include "ngraph/descriptor/tensor_view.hpp"
#include "ngraph/types/element_type.hpp"

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
    TensorViewWrapper(const std::shared_ptr<descriptor::TensorView>&);

    size_t get_size() const;
    const std::vector<size_t>& get_shape() const;
    const std::vector<size_t>& get_strides() const;
    const element::Type& get_element_type() const;
    const std::string& get_name() const;
    const std::string& get_type() const;
    bool is_output() const;

private:
    std::shared_ptr<descriptor::TensorView> m_tensor_view;
};
