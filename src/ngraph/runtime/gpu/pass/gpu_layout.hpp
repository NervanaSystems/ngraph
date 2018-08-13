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

#include "ngraph/pass/pass.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"

#define LAYOUT_DECL(op_type)                                                                       \
    layout<op_type>(ngraph::runtime::cpu::CPU_ExternalFunction * external_function,                \
                    std::shared_ptr<ngraph::Node> node)

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                using LayoutFunction =
                    std::function<void(CPU_ExternalFunction*, std::shared_ptr<ngraph::Node>)>;

                using LayoutOpMap = std::unordered_map<std::type_index, LayoutFunction>;

                class CPULayout : public ngraph::pass::CallGraphPass
                {
                public:
                    CPULayout(CPU_ExternalFunction* external_function)
                        : m_external_function(external_function)
                    {
                    }
                    virtual bool
                        run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes) override;

                    template <typename OP>
                    static void
                        layout(ngraph::runtime::cpu::CPU_ExternalFunction* external_function,
                               std::shared_ptr<ngraph::Node> node);

                private:
                    CPU_ExternalFunction* m_external_function;
                    static std::shared_ptr<Node> insert_input_conversions(
                        CPU_ExternalFunction* external_function,
                        std::shared_ptr<Node>& node,
                        const std::vector<mkldnn::memory::desc>& required_mds);
                    static void
                        set_output_layouts(std::shared_ptr<Node>& node,
                                           const std::vector<mkldnn::memory::desc>& output_mds);
                    static void set_native_layouts(CPU_ExternalFunction* external_function,
                                                   std::shared_ptr<Node> node,
                                                   bool use_replace);
                };
            }
        }
    }
}
