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

#include <memory>
#include <set>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "static_value_elimination.hpp"

using namespace ngraph;

template <typename T>
void copy_static_value_data(void* dst, const size_t* src, size_t n_items)
{
    T* dst_t = reinterpret_cast<T*>(dst);

    for (size_t i = 0; i < n_items; i++)
    {
        dst_t[i] = T(src[i]);
    }
}

void get_static_value_data(const StaticValue& sv, const element::Type& et, void** data_ptr)
{
    size_t buf_size = sv.size() * et.size();
    *data_ptr = ngraph::aligned_alloc(et.size(), buf_size);

    if (et == element::u64)
    {
        std::memcpy(*data_ptr, sv.data(), buf_size);
    }
    else if (et == element::f64)
    {
        copy_static_value_data<double>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::f32)
    {
        copy_static_value_data<float>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::i64)
    {
        copy_static_value_data<int64_t>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::i32)
    {
        copy_static_value_data<int32_t>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::i16)
    {
        copy_static_value_data<int16_t>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::i8)
    {
        copy_static_value_data<int8_t>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::u32)
    {
        copy_static_value_data<uint32_t>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::u16)
    {
        copy_static_value_data<uint16_t>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::u8)
    {
        copy_static_value_data<uint8_t>(*data_ptr, sv.data(), sv.size());
    }
    else if (et == element::boolean)
    {
        copy_static_value_data<char>(*data_ptr, sv.data(), sv.size());
    }
    else
    {
        NGRAPH_FAIL() << "get_static_value_data: Unknown element type " << et;
    }
}

bool ngraph::pass::StaticValueElimination::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;

    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter() || n->is_constant() || n->get_output_size() != 1)
        {
            continue;
        }

        if (!n->get_outputs()[0].has_static_value())
        {
            NGRAPH_DEBUG << " No static value, skipping " << *n;
            continue;
        }

        const StaticValue& sv = n->get_output_static_value(0);

        void* data_ptr;
        get_static_value_data(sv, n->get_output_element_type(0), &data_ptr);

        auto new_node = std::make_shared<op::Constant>(
            n->get_output_element_type(0), Shape{sv.size()}, data_ptr);

        std::free(data_ptr);

        replaced = true;
        NGRAPH_DEBUG << " Replacing statically evaluated node " << n->get_name() << " with "
                     << new_node->get_name();
        replace_node(n, new_node);
    }

    return replaced;
}
