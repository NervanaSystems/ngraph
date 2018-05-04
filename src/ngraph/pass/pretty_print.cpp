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

#include <fstream>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/pretty_print.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

bool pass::PrettyPrint::run_on_module(vector<shared_ptr<ngraph::Function>>& functions)
{
    for (shared_ptr<Function> f : functions)
    {
        m_stream << "Function " << f->get_name() << std::endl;
        m_stream << "======================" << std::endl;

        for (auto node : f->get_ordered_ops())
        {
            m_stream << node->get_name() << " = ";
            m_stream << node->description();

            bool first = true;
            for (auto kv : node->get_attribute_map())
            {
                if (first)
                {
                    m_stream << " [";
                }

                if (!first)
                {
                    m_stream << ", ";
                }

                m_stream << kv.first << "=" << kv.second->to_string();
                first = false;
            }

            if (!first)
            {
                m_stream << "]";
            }

            first = true;
            for (auto arg : node->get_arguments())
            {
                if (first)
                {
                    m_stream << " (";
                }

                if (!first)
                {
                    m_stream << ", ";
                }
                m_stream << arg->get_name();
                first = false;
            }

            if (!first)
            {
                m_stream << ")";
            }

            m_stream << std::endl;
        }
    }

    return false;
}
