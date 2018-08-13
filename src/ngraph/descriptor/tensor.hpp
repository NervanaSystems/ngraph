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

#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include "ngraph/type/type.hpp"

namespace ngraph
{
    class Node;

    namespace element
    {
        class Type;
    }

    namespace descriptor
    {
        class TensorView;
        class PrimaryTensorView;
        class Tensor;
    }
}

class ngraph::descriptor::Tensor
{
    friend class PrimaryTensorView;

private:
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(const element::Type& element_type,
           PrimaryTensorView* tensor_view,
           const std::string& name);

    std::string get_next_view_name();

public:
    const std::string& get_name() const { return m_name; }
    void set_pool_offset(size_t);
    size_t get_pool_offset() const;
    const element::Type& get_element_type() const { return m_element_type; }
    static std::string make_tensor_name(const Node* node, size_t value_index);
    PrimaryTensorView* get_primary_tensor_view() const { return m_primary_tensor_view; }
protected:
    const element::Type m_element_type;
    PrimaryTensorView* m_primary_tensor_view;
    std::string m_name;
    size_t m_next_view_id;
    size_t m_pool_offset;
};

std::ostream& operator<<(std::ostream&, const ngraph::descriptor::Tensor&);
