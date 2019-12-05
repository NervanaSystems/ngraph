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

#include "ngraph/opsets/opset.hpp"
#include "ngraph/ops.hpp"

std::mutex& ngraph::OpSet::get_mutex()
{
    static std::mutex opset_mutex;
    return opset_mutex;
}

ngraph::Node* ngraph::OpSet::create(const std::string& name) const
{
    auto type_info_it = m_name_type_info_map.find(name);
    return type_info_it == m_name_type_info_map.end()
               ? nullptr
               : FactoryRegistry<Node>::get().create(type_info_it->second);
}

const ngraph::OpSet& ngraph::get_opset0()
{
    static std::mutex init_mutex;
    static OpSet opset;
    if (opset.size() == 0)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (opset.size() == 0)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset0_tbl.hpp"
#undef NGRAPH_OP
        }
    }
    return opset;
}

const ngraph::OpSet& ngraph::get_opset1()
{
    static std::mutex init_mutex;
    static OpSet opset;
    if (opset.size() == 0)
    {
        std::lock_guard<std::mutex> guard(init_mutex);
        if (opset.size() == 0)
        {
#define NGRAPH_OP(NAME, NAMESPACE) opset.insert<NAMESPACE::NAME>();
#include "ngraph/opsets/opset1_tbl.hpp"
#undef NGRAPH_OP
        }
    }
    return opset;
}
