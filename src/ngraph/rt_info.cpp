//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/rt_info.hpp"
#include "ngraph/node.hpp"
#include "ngraph/variant.hpp"

namespace
{
    ngraph::Node::RTMap merge_runtime_info(const ngraph::NodeVector& nodes)
    {
        ngraph::Node::RTMap merged_info;
        for (auto& node : nodes)
        {
            for (auto& item : node->get_rt_info())
            {
                merged_info[item.first] = item.second;
            }
        }

        ngraph::Node::RTMap new_info;
        for (auto& item : merged_info)
        {
            if (auto merge_attr = item.second->merge(nodes))
            {
                new_info[item.second->get_type_info().name] = merge_attr;
            }
        }

        return new_info;
    }
}

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, std::shared_ptr<ngraph::Node> to)
{
    auto& rt_info_from = from->get_rt_info();
    auto& rt_info_to = to->get_rt_info();
    rt_info_to = rt_info_from;
}

void ngraph::copy_runtime_info(std::shared_ptr<ngraph::Node> from, ngraph::NodeVector to)
{
    for (auto& op : to)
    {
        copy_runtime_info(from, op);
    }
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, std::shared_ptr<ngraph::Node> to)
{
    auto& rt_info_to = to->get_rt_info();
    rt_info_to = merge_runtime_info(from);
}

void ngraph::copy_runtime_info(const ngraph::NodeVector& from, ngraph::NodeVector to)
{
    auto merged_info = merge_runtime_info(from);
    for (auto& node : to)
    {
        auto& rt_info_to = node->get_rt_info();
        rt_info_to = merged_info;
    }
}
